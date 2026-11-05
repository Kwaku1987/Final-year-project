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
| Training Mode is an explicit, audited step | Mode selection is deliberate, and every entry into Training Mode is written to the audit trail (`training_mode_entered`) |
| Lecturer mode never sees the dataset | Assessment Mode loads only the saved model bundle; the training data is never required |
| No training data written to disk | Uploads live in the Streamlit session only; only the model bundle is persisted |
| Destructive actions require confirmation | Model deletion needs an explicit checkbox |
| Methodology caveats travel with results | Every report carries `report.caveats` |

### There is no administrator password

Training Mode used to be behind a shared password (a SHA-256 digest in
`SPAS_ADMIN_PASSWORD_HASH`). That gate was removed at the owner's request, and
the code that implemented it (`utils/auth.py`) was deleted rather than left
disabled, so there is no dead path that looks like protection.

What that means, stated plainly:

* **Anyone who can load the app can enter Training Mode**, and therefore upload a
  dataset, retrain, and delete the production bundle. There is no attempt limit
  and no credential to get wrong.
* **The audit trail still records it.** Entering Training Mode writes
  `training_mode_entered`; loading a dataset, saving and deleting a model are
  audited too. Use is visible even though it is not restricted.
* **A public URL is a demo, not the departmental system.** For real use, publish
  Assessment Mode only, or put the app behind your institution's login, a VPN, or
  a reverse proxy with authentication.
* **Nothing else was weakened:** CORS and XSRF protection stay on, tracebacks stay
  hidden (`showErrorDetails = false`), uploads are capped at 25 MB, and no student
  data is written to disk — uploads live in the session, and only the model bundle
  is persisted.

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
   with authentication) — Streamlit itself serves plain HTTP and **the app has no
   login of its own**, so nothing else stands in front of Training Mode.
2. Bind to localhost or an internal network; do not expose the port publicly.
3. Keep configuration out of the repository (`.env`, `.streamlit/secrets.toml`);
   there is no credential inside the app to manage or rotate.
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
