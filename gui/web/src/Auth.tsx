import { FormEvent, useEffect, useState } from "react";
import {
  AlertCircle,
  Apple,
  Chrome,
  KeyRound,
  Loader2,
  LogIn,
  LogOut,
  RefreshCw,
  ShieldCheck,
  Trash2,
  UserPlus
} from "lucide-react";
import { AuthStatus, AuthUser, api, setApiToken } from "./api";

const PROVIDER_LABEL: Record<string, string> = { google: "Google", apple: "Apple" };
const PROVIDER_ICON: Record<string, JSX.Element> = {
  google: <Chrome size={16} />,
  apple: <Apple size={16} />
};

/** Full-screen sign-in gate shown when auth is required and there is no session. */
export function LoginScreen({ status, onAuthenticated }: { status: AuthStatus; onAuthenticated: () => void }) {
  const [step, setStep] = useState<"password" | "twofa">("password");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submitPassword = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await api.login(username.trim(), password);
      if (result.needs_2fa) {
        setStep("twofa");
      } else {
        if (result.token) setApiToken(result.token);
        onAuthenticated();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign in failed.");
    } finally {
      setBusy(false);
    }
  };

  const submit2fa = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await api.login2fa(username.trim(), code.trim());
      if (result.token) setApiToken(result.token);
      onAuthenticated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid code.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-backdrop">
      <div className="login-card">
        <div className="login-brand">
          <span className="login-logo">S</span>
          <div>
            <strong>SNDR Control Center</strong>
            <small>Sign in to continue</small>
          </div>
        </div>

        {error && (
          <div className="login-error">
            <AlertCircle size={15} /> <span>{error}</span>
          </div>
        )}

        {step === "password" ? (
          <form className="login-form" onSubmit={submitPassword}>
            <label className="field">
              <span>Username</span>
              <input
                autoFocus
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                placeholder={status.context.system_user}
                autoComplete="username"
              />
            </label>
            <label className="field">
              <span>Password</span>
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="current-password"
              />
            </label>
            <button className="login-go" type="submit" disabled={busy || !username || !password}>
              {busy ? <Loader2 size={16} className="spin" /> : <LogIn size={16} />}
              {busy ? "Signing in…" : "Sign in"}
            </button>
          </form>
        ) : (
          <form className="login-form" onSubmit={submit2fa}>
            <p className="login-hint">Enter the 6-digit code from your authenticator app.</p>
            <label className="field">
              <span>Authentication code</span>
              <input
                autoFocus
                value={code}
                onChange={(event) => setCode(event.target.value.replace(/\D/g, "").slice(0, 6))}
                inputMode="numeric"
                placeholder="000000"
              />
            </label>
            <button className="login-go" type="submit" disabled={busy || code.length < 6}>
              {busy ? <Loader2 size={16} className="spin" /> : <ShieldCheck size={16} />}
              {busy ? "Verifying…" : "Verify"}
            </button>
            <button type="button" className="link-button" onClick={() => { setStep("password"); setCode(""); }}>
              Back
            </button>
          </form>
        )}

        {step === "password" && status.oauth_providers.length > 0 && (
          <div className="login-oauth">
            <div className="login-divider"><span>or</span></div>
            {status.oauth_providers.map((provider) => (
              <a key={provider} className="oauth-button" href={api.oauthLoginUrl(provider)}>
                {PROVIDER_ICON[provider] ?? <KeyRound size={16} />}
                Continue with {PROVIDER_LABEL[provider] ?? provider}
              </a>
            ))}
          </div>
        )}

        <div className="login-context">
          {status.context.in_container ? "Container deployment" : "Host deployment"}
          {" · "}backends: {status.backends.join(", ")}
        </div>
      </div>
    </div>
  );
}

/** Header chip showing the current account with a logout action. */
export function AccountMenu({ user, onLoggedOut }: { user: AuthUser; onLoggedOut: () => void }) {
  const [busy, setBusy] = useState(false);
  const logout = async () => {
    setBusy(true);
    try {
      await api.logout();
    } catch {
      // ignore — clear client state regardless
    }
    setApiToken("");
    onLoggedOut();
  };
  return (
    <div className="account-menu">
      <div className="account-id">
        <span className="account-avatar">{user.username.slice(0, 1).toUpperCase()}</span>
        <div>
          <strong>{user.username}</strong>
          <small>{user.role}{user.totp_enabled ? " · 2FA" : ""}</small>
        </div>
      </div>
      <button className="tool-button" onClick={logout} disabled={busy} title="Sign out">
        <LogOut size={15} /> Sign out
      </button>
    </div>
  );
}

/** Current-user security: change password + enrol/disable 2FA. */
export function SecurityPanel({ user, onChanged }: { user: AuthUser; onChanged: () => void }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [pwMsg, setPwMsg] = useState<string | null>(null);
  const [enroll, setEnroll] = useState<{ secret: string; otpauth_uri: string } | null>(null);
  const [code, setCode] = useState("");
  const [twoMsg, setTwoMsg] = useState<string | null>(null);
  const [recoveryCodes, setRecoveryCodes] = useState<string[] | null>(null);

  const changePassword = async (event: FormEvent) => {
    event.preventDefault();
    setPwMsg(null);
    try {
      await api.changePassword(current, next);
      setPwMsg("Password updated.");
      setCurrent("");
      setNext("");
    } catch (err) {
      setPwMsg(err instanceof Error ? err.message : "Failed.");
    }
  };

  const startEnroll = async () => {
    setTwoMsg(null);
    try {
      setEnroll(await api.enroll2fa());
    } catch (err) {
      setTwoMsg(err instanceof Error ? err.message : "Failed.");
    }
  };
  const activate = async () => {
    setTwoMsg(null);
    try {
      const res = await api.activate2fa(code.trim());
      setRecoveryCodes(res.recovery_codes);
      setEnroll(null);
      setCode("");
      onChanged();
    } catch (err) {
      setTwoMsg(err instanceof Error ? err.message : "Failed.");
    }
  };
  const regenerate = async () => {
    setTwoMsg(null);
    try {
      setRecoveryCodes((await api.regenerateRecovery()).recovery_codes);
    } catch (err) {
      setTwoMsg(err instanceof Error ? err.message : "Failed.");
    }
  };
  const disable = async () => {
    setTwoMsg(null);
    setRecoveryCodes(null);
    try {
      await api.disable2fa();
      onChanged();
    } catch (err) {
      setTwoMsg(err instanceof Error ? err.message : "Failed.");
    }
  };
  const revokeAll = async () => {
    try {
      await api.revokeSessions();
      setApiToken("");
      onChanged();
    } catch (err) {
      setTwoMsg(err instanceof Error ? err.message : "Failed.");
    }
  };

  return (
    <div className="security-panel">
      <div className="security-block">
        <h4><KeyRound size={15} /> Change password</h4>
        {user.has_password ? (
          <form className="login-form" onSubmit={changePassword}>
            <label className="field">
              <span>Current password</span>
              <input type="password" value={current} onChange={(event) => setCurrent(event.target.value)} />
            </label>
            <label className="field">
              <span>New password (min 8 chars)</span>
              <input type="password" value={next} onChange={(event) => setNext(event.target.value)} />
            </label>
            <button className="primary-button" type="submit" disabled={!current || next.length < 8}>Update password</button>
            {pwMsg && <p className="muted">{pwMsg}</p>}
          </form>
        ) : (
          <p className="muted">This account signs in via {user.source} — no local password.</p>
        )}
      </div>

      <div className="security-block">
        <h4><ShieldCheck size={15} /> Two-factor authentication</h4>
        {recoveryCodes ? (
          <div className="recovery-codes">
            <p className="recovery-warn"><AlertCircle size={14} /> Save these recovery codes now — shown only once. Each works once if you lose your authenticator.</p>
            <div className="recovery-grid">
              {recoveryCodes.map((rc) => <code key={rc}>{rc}</code>)}
            </div>
            <button className="ghost-button" onClick={() => setRecoveryCodes(null)}>I saved them</button>
          </div>
        ) : user.totp_enabled ? (
          <>
            <p className="muted">2FA is <strong>enabled</strong> · {user.recovery_codes_remaining} recovery codes left.</p>
            <div className="security-actions">
              <button className="ghost-button" onClick={regenerate}>Regenerate recovery codes</button>
              <button className="ghost-button" onClick={disable}>Disable 2FA</button>
            </div>
          </>
        ) : enroll ? (
          <div className="twofa-enroll">
            <p className="muted">Add this secret to your authenticator app (manual entry), then enter the code to confirm.</p>
            <div className="twofa-secret">{enroll.secret}</div>
            <code className="twofa-uri">{enroll.otpauth_uri}</code>
            <label className="field">
              <span>6-digit code</span>
              <input value={code} onChange={(event) => setCode(event.target.value.replace(/\D/g, "").slice(0, 6))} inputMode="numeric" placeholder="000000" />
            </label>
            <button className="primary-button" onClick={activate} disabled={code.length < 6}>Activate 2FA</button>
          </div>
        ) : (
          <>
            <p className="muted">Protect this account with a time-based one-time code.</p>
            <button className="primary-button" onClick={startEnroll}><ShieldCheck size={14} /> Enable 2FA</button>
          </>
        )}
        {twoMsg && <p className="muted">{twoMsg}</p>}
      </div>

      <div className="security-block">
        <h4><LogOut size={15} /> Sessions</h4>
        <p className="muted">Invalidate every active session for this account (including other devices).</p>
        <button className="ghost-button" onClick={revokeAll}>Sign out everywhere</button>
      </div>
    </div>
  );
}

/** Admin-only user directory: list, create, delete accounts. */
export function UserAdminPanel({ currentUser }: { currentUser: AuthUser }) {
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("operator");

  const load = async () => {
    try {
      setUsers((await api.listUsers()).users);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load users.");
    }
  };
  useEffect(() => { void load(); }, []);

  const create = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      await api.createUser(username.trim(), password, role);
      setUsername("");
      setPassword("");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create user.");
    }
  };
  const remove = async (name: string) => {
    setError(null);
    try {
      await api.deleteUser(name);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete user.");
    }
  };

  return (
    <div className="user-admin">
      {error && <div className="login-error"><AlertCircle size={14} /> <span>{error}</span></div>}
      <div className="user-admin-head">
        <span>{users.length} account{users.length === 1 ? "" : "s"}</span>
        <button className="ghost-button" onClick={() => void load()}><RefreshCw size={13} /> Refresh</button>
      </div>
      <table className="module-table">
        <thead>
          <tr><th>User</th><th>Role</th><th>Source</th><th>2FA</th><th /></tr>
        </thead>
        <tbody>
          {users.map((u) => (
            <tr key={u.username}>
              <td><strong>{u.username}</strong>{u.email ? <small> · {u.email}</small> : null}</td>
              <td>{u.role}</td>
              <td>{u.source}</td>
              <td>{u.totp_enabled ? "on" : "—"}</td>
              <td className="preset-row-actions">
                {u.username !== currentUser.username && (
                  <button className="icon-button" title={`Delete ${u.username}`} onClick={() => void remove(u.username)}>
                    <Trash2 size={14} />
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <form className="user-create" onSubmit={create}>
        <h4><UserPlus size={15} /> Create user</h4>
        <div className="user-create-row">
          <input value={username} onChange={(event) => setUsername(event.target.value)} placeholder="username" />
          <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="password (min 8)" />
          <select value={role} onChange={(event) => setRole(event.target.value)}>
            <option value="operator">operator</option>
            <option value="admin">admin</option>
            <option value="viewer">viewer</option>
          </select>
          <button className="primary-button" type="submit" disabled={!username || password.length < 8}>Create</button>
        </div>
      </form>
    </div>
  );
}
