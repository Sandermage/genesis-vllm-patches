// SPDX-License-Identifier: Apache-2.0
// Lightweight EN/RU i18n. A `t(lang, key)` lookup over a flat dictionary plus a
// `useLang()` hook that persists the choice and broadcasts changes so every
// component re-renders on a language switch. New surfaces (the Virtualization
// panel, the sidebar nav) are authored bilingually; existing screens fall back
// to English until they adopt `t()`, so adoption is incremental and safe.
import { useEffect, useState } from "react";

export type Lang = "en" | "ru";
const LANG_KEY = "sndr.gui.lang";
const EVT = "sndr-lang-change";

export function getLang(): Lang {
  if (typeof window === "undefined") return "en";
  const v = window.localStorage.getItem(LANG_KEY);
  return v === "ru" ? "ru" : "en";
}

export function setLang(lang: Lang): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LANG_KEY, lang);
  window.dispatchEvent(new CustomEvent(EVT, { detail: lang }));
}

export function useLang(): [Lang, (l: Lang) => void] {
  const [lang, setLangState] = useState<Lang>(getLang);
  useEffect(() => {
    const onChange = () => setLangState(getLang());
    window.addEventListener(EVT, onChange);
    window.addEventListener("storage", onChange); // cross-tab
    return () => { window.removeEventListener(EVT, onChange); window.removeEventListener("storage", onChange); };
  }, []);
  return [lang, setLang];
}

type Dict = Record<string, string>;

const EN: Dict = {
  // nav
  "nav.dashboard": "Overview", "nav.fleet": "Fleet", "nav.containers": "Containers",
  "nav.kubernetes": "Kubernetes", "nav.virtualization": "Virtualization", "nav.hardware": "Hardware",
  "nav.setup": "Setup", "nav.models": "Models", "nav.presets": "Presets", "nav.configs": "Configs",
  "nav.planner": "Planner", "nav.launch-plan": "Launch Plan", "nav.services": "Services",
  "nav.chat": "Chat & Copilot", "nav.routing": "Routing", "nav.doctor": "Doctor", "nav.patches": "Patches",
  "nav.evidence": "Evidence", "nav.advanced": "Advanced",
  // common
  "common.refresh": "Refresh", "common.online": "online", "common.offline": "offline",
  "common.running": "running", "common.stopped": "stopped", "common.nodes": "nodes",
  "common.connect": "Connect", "common.loading": "Loading…", "common.none": "none",
  "common.cpu": "CPU", "common.memory": "Memory", "common.disk": "Disk", "common.uptime": "Uptime",
  // virtualization
  "virt.title": "Virtualization",
  "virt.subtitle": "One pane over your compute — Proxmox VE hosts & guests, KubeVirt VMs, and Kubernetes nodes — linked back to the SNDR presets they run.",
  "virt.proxmox": "Proxmox VE", "virt.kubevirt": "KubeVirt", "virt.k8sNodes": "K8s nodes",
  "virt.nodes": "Nodes", "virt.pods": "Pods", "virt.events": "Events", "virt.deploy": "Deploy",
  "virt.hosts": "Hosts", "virt.guests": "Guests", "virt.vms": "VMs", "virt.lxc": "LXC",
  "virt.vm": "VM", "virt.container": "Container",
  "virt.proxmoxNotConfigured": "Proxmox not connected",
  "virt.proxmoxConnectHelp": "Set SNDR_PROXMOX_HOST, SNDR_PROXMOX_TOKEN_ID and SNDR_PROXMOX_TOKEN_SECRET on the daemon to monitor your Proxmox cluster — host nodes, VMs and LXC with their resources, plus the SNDR preset each guest hosts.",
  "virt.kubevirtNotInstalled": "KubeVirt is not installed on this cluster.",
  "virt.kubevirtHelp": "KubeVirt runs VMs as first-class Kubernetes objects. Install the KubeVirt operator to manage VMs alongside pods here.",
  "virt.k8sNotConnected": "No Kubernetes cluster connected — see the Kubernetes tab.",
  "virt.noGuests": "No VMs or containers on this Proxmox cluster yet.",
  "virt.sndrManaged": "SNDR-managed", "virt.preset": "preset", "virt.node": "node", "virt.tags": "tags",
  "virt.value": "What it gives you",
  "virt.valueBody": "Your GPU engines run inside Proxmox VMs/LXC and (optionally) KubeVirt. This view connects each guest to the SNDR preset it hosts — so the infrastructure (where it runs) and the engine (what runs) are one story, the same way Containers and Kubernetes already are.",
  // virtualization — provider + tab explanations
  "virt.proxmoxAbout": "Proxmox VE is your bare-metal hypervisor. SNDR runs each GPU engine inside an LXC container or a VM with the GPU passed through. Connect read-only to watch nodes & guests; use Create to generate a ready-to-run provision script for a new one.",
  "virt.k8sAbout": "Kubernetes schedules engines as pods across your GPU nodes. Watch nodes, pods and events, see KubeVirt VMs, and use Deploy to render a Deployment manifest from a preset.",
  "virt.tabHostsHelp": "Physical Proxmox nodes — CPU, memory and disk pressure per host, online/offline.",
  "virt.tabGuestsHelp": "Every VM and LXC across the cluster, each linked to the SNDR preset it runs.",
  "virt.tabNodesHelp": "Cluster nodes — readiness, GPU capacity/allocation, kubelet version and taints.",
  "virt.tabPodsHelp": "Running pods with phase, restarts and GPU requests, across all namespaces.",
  "virt.tabEventsHelp": "Recent cluster events — warnings first, so scheduling/health issues surface fast.",
  // virtualization — Proxmox create
  "virt.create": "Create",
  "virt.pxCreateTitle": "Create a Proxmox guest",
  "virt.pxCreateBody": "Pick a SNDR preset and a guest type, then generate the exact provision script. Read-only here — nothing is created until you run that script on the Proxmox node (or apply it over SSH from the Install wizard).",
  "virt.guestType": "Guest type",
  "virt.pxLxc": "LXC container",
  "virt.pxVm": "Virtual machine",
  "virt.pxLxcHelp": "Lighter, shares the host kernel — the usual choice for a GPU engine with passthrough.",
  "virt.pxVmHelp": "Full isolation with its own kernel — when you need a separate OS or stricter boundaries.",
  "virt.generate": "Generate provision script",
  "virt.pxRunOnNode": "Run on the Proxmox node",
  "virt.pxApplySsh": "Apply over SSH (Install wizard)",
  "virt.copyScript": "Copy script",
  "virt.copied": "copied",
  // routing
  "rt.title": "Spec-decode workload routing",
  "rt.intro": "The deterministic router the gateway uses: it classifies each request by its shape — response_format, tool_choice, and workload_class — and resolves it to the bench-validated profile that serves that shape fastest, falling back safely when a profile denies it.",
  "rt.how": "How it decides", "rt.live": "Live engine", "rt.classifier": "Request classifier",
  "rt.howBody": "Each profile is bench-validated per workload class (free-chat, code-gen, tool-calls, structured-JSON, summarization, long-context) with a measured TPS delta vs baseline. A request's signals pick the class; the router routes to the allowed profile with the best delta, or the safe baseline when the spec-decode path is denied for that shape. Bench artifacts populate the per-workload table; the classifier below works live regardless.",
  "rt.classifierHelp": "Pick a request shape and see which profile it resolves to and why — the same decision the gateway makes.",
  "rt.liveHelp": "Live from the active engine's /metrics — what it is actually serving right now.",
};

const RU: Dict = {
  // nav
  "nav.dashboard": "Обзор", "nav.fleet": "Флот", "nav.containers": "Контейнеры",
  "nav.kubernetes": "Kubernetes", "nav.virtualization": "Виртуализация", "nav.hardware": "Железо",
  "nav.setup": "Установка", "nav.models": "Модели", "nav.presets": "Пресеты", "nav.configs": "Конфиги",
  "nav.planner": "Планировщик", "nav.launch-plan": "План запуска", "nav.services": "Сервисы",
  "nav.chat": "Чат и Copilot", "nav.routing": "Маршрутизация", "nav.doctor": "Диагностика", "nav.patches": "Патчи",
  "nav.evidence": "Доказательства", "nav.advanced": "Расширенное",
  // common
  "common.refresh": "Обновить", "common.online": "онлайн", "common.offline": "офлайн",
  "common.running": "работает", "common.stopped": "остановлен", "common.nodes": "ноды",
  "common.connect": "Подключить", "common.loading": "Загрузка…", "common.none": "нет",
  "common.cpu": "CPU", "common.memory": "Память", "common.disk": "Диск", "common.uptime": "Аптайм",
  // virtualization
  "virt.title": "Виртуализация",
  "virt.subtitle": "Единая панель по вычислениям — хосты и гости Proxmox VE, VM KubeVirt и ноды Kubernetes — связанные с пресетами SNDR, которые на них работают.",
  "virt.proxmox": "Proxmox VE", "virt.kubevirt": "KubeVirt", "virt.k8sNodes": "Ноды k8s",
  "virt.nodes": "Ноды", "virt.pods": "Поды", "virt.events": "События", "virt.deploy": "Деплой",
  "virt.hosts": "Хосты", "virt.guests": "Гости", "virt.vms": "VM", "virt.lxc": "LXC",
  "virt.vm": "VM", "virt.container": "Контейнер",
  "virt.proxmoxNotConfigured": "Proxmox не подключён",
  "virt.proxmoxConnectHelp": "Задай на демоне SNDR_PROXMOX_HOST, SNDR_PROXMOX_TOKEN_ID и SNDR_PROXMOX_TOKEN_SECRET, чтобы мониторить кластер Proxmox — хост-ноды, VM и LXC с ресурсами, плюс пресет SNDR на каждом госте.",
  "virt.kubevirtNotInstalled": "KubeVirt не установлен в этом кластере.",
  "virt.kubevirtHelp": "KubeVirt запускает VM как полноценные объекты Kubernetes. Установи оператор KubeVirt, чтобы управлять VM рядом с подами здесь.",
  "virt.k8sNotConnected": "Кластер Kubernetes не подключён — см. вкладку Kubernetes.",
  "virt.noGuests": "На этом кластере Proxmox пока нет VM или контейнеров.",
  "virt.sndrManaged": "Под управлением SNDR", "virt.preset": "пресет", "virt.node": "нода", "virt.tags": "теги",
  "virt.value": "Что это даёт",
  "virt.valueBody": "Твои GPU-движки живут внутри Proxmox VM/LXC и (опционально) KubeVirt. Эта панель связывает каждого гостя с пресетом SNDR, который на нём крутится — инфраструктура (где работает) и движок (что работает) становятся одной историей, как уже сделано для Контейнеров и Kubernetes.",
  // virtualization — пояснения по провайдерам и вкладкам
  "virt.proxmoxAbout": "Proxmox VE — твой гипервизор на железе. SNDR запускает каждый GPU-движок в LXC-контейнере или VM с пробросом GPU. Подключение только для чтения — следи за нодами и гостями; вкладка «Создать» генерирует готовый скрипт развёртывания нового гостя.",
  "virt.k8sAbout": "Kubernetes раскладывает движки как поды по GPU-нодам. Смотри ноды, поды и события, VM KubeVirt, а вкладка «Деплой» рендерит манифест Deployment из пресета.",
  "virt.tabHostsHelp": "Физические ноды Proxmox — загрузка CPU, памяти и диска по каждому хосту, онлайн/офлайн.",
  "virt.tabGuestsHelp": "Все VM и LXC по кластеру, каждый связан с пресетом SNDR, который на нём работает.",
  "virt.tabNodesHelp": "Ноды кластера — готовность, ёмкость/занятость GPU, версия kubelet и taints.",
  "virt.tabPodsHelp": "Работающие поды: фаза, рестарты и запросы GPU, по всем неймспейсам.",
  "virt.tabEventsHelp": "Свежие события кластера — предупреждения сверху, чтобы проблемы планирования/здоровья были видны сразу.",
  // virtualization — создание гостя Proxmox
  "virt.create": "Создать",
  "virt.pxCreateTitle": "Создать гостя Proxmox",
  "virt.pxCreateBody": "Выбери пресет SNDR и тип гостя, затем сгенерируй точный скрипт развёртывания. Здесь только чтение — ничего не создаётся, пока ты не запустишь скрипт на ноде Proxmox (или не применишь по SSH из мастера установки).",
  "virt.guestType": "Тип гостя",
  "virt.pxLxc": "LXC-контейнер",
  "virt.pxVm": "Виртуальная машина",
  "virt.pxLxcHelp": "Легче, использует ядро хоста — обычный выбор для GPU-движка с пробросом.",
  "virt.pxVmHelp": "Полная изоляция со своим ядром — когда нужна отдельная ОС или строгие границы.",
  "virt.generate": "Сгенерировать скрипт",
  "virt.pxRunOnNode": "Запусти на ноде Proxmox",
  "virt.pxApplySsh": "Применить по SSH (мастер установки)",
  "virt.copyScript": "Копировать скрипт",
  "virt.copied": "скопировано",
  // routing
  "rt.title": "Маршрутизация по типам нагрузки (spec-decode)",
  "rt.intro": "Детерминированный маршрутизатор, который использует шлюз: он классифицирует каждый запрос по его форме — response_format, tool_choice и workload_class — и направляет на bench-валидированный профиль, который обслуживает эту форму быстрее всего, безопасно откатываясь, когда профиль её отклоняет.",
  "rt.how": "Как принимается решение", "rt.live": "Живой движок", "rt.classifier": "Классификатор запросов",
  "rt.howBody": "Каждый профиль bench-валидирован по классам нагрузки (free-chat, code-gen, tool-calls, structured-JSON, суммаризация, long-context) с измеренной дельтой TPS относительно базы. Сигналы запроса выбирают класс; маршрутизатор направляет на разрешённый профиль с лучшей дельтой, либо на безопасную базу, когда spec-decode-путь отклонён для этой формы. Bench-артефакты наполняют таблицу по нагрузкам; классификатор ниже работает вживую в любом случае.",
  "rt.classifierHelp": "Выбери форму запроса и увидь, в какой профиль он разрешается и почему — то же решение, что принимает шлюз.",
  "rt.liveHelp": "Вживую из /metrics активного движка — что он реально обслуживает прямо сейчас.",
};

const DICT: Record<Lang, Dict> = { en: EN, ru: RU };

export function t(lang: Lang, key: string, fallback?: string): string {
  return DICT[lang]?.[key] ?? DICT.en[key] ?? fallback ?? key;
}
