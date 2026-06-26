# SPDX-License-Identifier: Apache-2.0
"""Tests for the comprehensive GPU + hardware telemetry collector."""
from sndr.product_api.legacy import gpu_telemetry as G

_ROW = (
    "NVIDIA RTX A5000, GPU-abc, 1320621012345, 535.104.05, 94.02.5c, "
    "12000, 24564, 12564, 45, 30, 62, [N/A], "
    "145.5, 230, 230, 100, 41, 4, 4, 16, 16, "
    "1695, 1695, 7000, 8000, 1695, Default, P2, 0, 0"
)


def test_parse_gpu_csv_maps_and_cleans():
    gpus = G.parse_gpu_csv(_ROW + "\n" + _ROW)  # two identical GPUs
    assert len(gpus) == 2
    g = gpus[0]
    assert g["name"] == "NVIDIA RTX A5000"
    assert g["serial"] == "1320621012345"
    assert g["gpu_util"] == 45 and g["temp_gpu"] == 62
    assert g["power"] == 145.5 and g["power_max_limit"] == 230.0
    assert g["power_min_limit"] == 100.0
    assert g["mem_used"] == 12000.0 and g["mem_total"] == 24564.0
    assert g["pcie_gen"] == 4 and g["pcie_width"] == 16
    assert g["clock_gpu"] == 1695 and g["clock_mem_max"] == 8000
    assert g["pstate"] == "P2" and g["ecc_corrected"] == "0"
    # [N/A] cleaned to None
    assert g["temp_mem"] is None


def test_parse_gpu_csv_skips_short_rows():
    assert G.parse_gpu_csv("too, few, cells") == []
    assert G.parse_gpu_csv("") == []


def test_int_field_tolerates_float_and_garbage():
    assert G._i("1695.0") == 1695
    assert G._i("") is None
    assert G._i("[N/A]") is None
    assert G._f("145.5") == 145.5
    assert G._f("x") is None


def test_parse_system():
    s = G.parse_system(
        "processor\t: 0\nmodel name\t: AMD Ryzen 9 5950X\n",
        "MemTotal: 65536000 kB\nMemAvailable: 32768000 kB\n",
        hostname="gpu-build-01", cpu_count=32,
    )
    assert s["cpu"] == "AMD Ryzen 9 5950X"
    assert s["hostname"] == "gpu-build-01" and s["cpu_count"] == 32
    assert s["ram_total_gb"] == 62.5
    assert s["ram_used_gb"] == 31.2


_NETDEV = (
    "Inter-|   Receive                                                |  Transmit\n"
    " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n"
    "    lo: 1000 10 0 0 0 0 0 0 1000 10 0 0 0 0 0 0\n"
    "  eth0: 5000000000 40 0 0 0 0 0 0 2000000000 30 0 0 0 0 0 0\n"
    "  eth1: 100 1 0 0 0 0 0 0 200 2 0 0 0 0 0 0\n"
)


def test_parse_net():
    net = G.parse_net(_NETDEV, "192.0.2.10 172.17.0.1\n")
    names = [i["name"] for i in net["interfaces"]]
    assert "lo" not in names              # loopback dropped
    assert names[0] == "eth0"             # busiest first
    assert net["interfaces"][0]["rx_bytes"] == 5000000000
    assert net["interfaces"][0]["tx_bytes"] == 2000000000
    assert net["primary_ip"] == "192.0.2.10"


def test_parse_net_empty():
    assert G.parse_net("", "") == {"interfaces": [], "primary_ip": None}


def test_parse_disk():
    df = (
        "Filesystem     1024-blocks      Used  Available Capacity Mounted on\n"
        "/dev/sda1        209715200 104857600  104857600      50% /\n"
    )
    d = G.parse_disk(df)
    assert d["mount"] == "/"
    assert d["total_gb"] == 200.0 and d["free_gb"] == 100.0
    assert d["used_pct"] == 50.0


def test_parse_disk_garbage():
    assert G.parse_disk("") is None
    assert G.parse_disk("only one line\n") is None


def test_collect_with_runner():
    """collect() drives a runner; GPU + system + net + disk all populate."""
    def run(argv):
        cmd = argv[0]
        if cmd == "nvidia-smi":
            return 0, _ROW, ""
        if argv[:2] == ["cat", "/proc/cpuinfo"]:
            return 0, "model name\t: Test CPU\n", ""
        if argv[:2] == ["cat", "/proc/meminfo"]:
            return 0, "MemTotal: 1048576 kB\nMemAvailable: 524288 kB\n", ""
        if argv[:2] == ["cat", "/proc/net/dev"]:
            return 0, _NETDEV, ""
        if argv == ["hostname", "-I"]:
            return 0, "10.0.0.5\n", ""
        if cmd == "hostname":
            return 0, "node-1\n", ""
        if cmd == "nproc":
            return 0, "8\n", ""
        if cmd == "uname":
            return 0, "Linux 6.8.0 x86_64\n", ""
        if cmd == "df":
            return 0, "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/sda1 209715200 104857600 104857600 50% /\n", ""
        return 127, "", "unknown"

    t = G.collect(run)
    assert t.error is None
    assert len(t.gpus) == 1 and t.gpus[0]["name"] == "NVIDIA RTX A5000"
    assert t.gpus[0]["serial"] == "1320621012345"
    assert t.system["cpu"] == "Test CPU" and t.system["cpu_count"] == 8
    assert t.system["ram_total_gb"] == 1.0
    assert t.system["platform"] == "Linux 6.8.0 x86_64"
    assert t.system["primary_ip"] == "10.0.0.5"
    assert t.system["net"][0]["name"] == "eth0"
    assert t.system["disk"]["free_gb"] == 100.0


def test_collect_reports_error_when_no_gpu():
    def run(argv):
        if argv[0] == "nvidia-smi":
            return 127, "", "nvidia-smi: not found"
        return 0, "", ""

    t = G.collect(run)
    assert t.gpus == ()
    assert t.error and "not found" in t.error
