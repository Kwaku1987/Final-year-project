# Data Protection & Access Control

Student marks, attendance and identifiers are **personal data**. In Ghana the
governing instrument is the **Data Protection Act, 2012 (Act 843)**; if any
subject is in the EU/UK, GDPR-equivalent obligations also apply. This page states
what the system does, what it deliberately does not do, and what the deploying
institution must add.

---

## 1. What the system now does

| Control | Implementation |
|---|---|
| Training Mode is password-gated | `utils/auth.py` + `backend/config.py` (`ADMIN_PASSWORD_HASH`) |
| Lecturer mode never sees the dataset | Assessment Mode loads only the saved model bundle; the training data is never required |
| No training data written to disk | Uploads live in the Streamlit session only; only the model bundle is persisted |
| Destructive actions require confirmation | Model deletion needs an explicit checkbox |
| Methodology caveats travel with results | Every report carries `report.caveats` |

### Setting the administrator password

```bash
python -m utils.auth hash "your-strong-password"
export SPAS_ADMIN_PASSWORD_HASH="<the printed 64-char digest>"
streamlit run app.py
```

If `SPAS_ADMIN_PASSWORD_HASH` is unset the app falls back to the documented
development password `change-me-admin` and says so on screen. **Never deploy
with the fallback.**

Five failed attempts per session lock Training Mode until the page is reloaded.
Comparison is constant-time (`hmac.compare_digest`).

---

## 2. What this is NOT

* **Not authentication of individuals.** It is one shared secret for a demo. No
  user accounts, no per-user attribution.
* **Not an audit log.** Actions are logged to the server console via Python
  `logging`, not to a tamper-evident store.
* **Not encryption.** The model bundle and any exported PDF/Excel files are
  plain files.
* **Not a retention policy.** Nothing is deleted automatically.

---

## 3. Deployment checklist for a real institution

1. Put the app behind institutional SSO (or at minimum HTTPS + a reverse proxy
   with authentication) — Streamlit itself serves plain HTTP.
2. Bind to localhost or an internal network; do not expose the port publicly.
3. Set `SPAS_ADMIN_PASSWORD_HASH`; rotate it; store it in a secret manager, never
   in the repository.
4. Write an audit log of who trained, saved, deleted or exported what (append
   `user`, `timestamp`, `action`, `model_version` to a durable store).
5. Decide and document the **lawful basis** and **retention period** for student
   data; delete datasets after the project's assessment period.
6. Anonymise or pseudonymise before any analysis that leaves the institution.
7. Record a Data Protection Impact Assessment if the tool informs decisions
   about students (it is a decision-support tool, so this is likely required).
8. State plainly, in the student-facing interface, that the output is
   **advisory and non-binding**.

---

## 4. Reporting a concern

If you find a security issue in this codebase, open a private advisory on the
repository rather than a public issue. Do **not** paste credentials, tokens or
student data into issues, chat or commit messages — tokens committed once remain
in the Git history and must be revoked, not deleted.
