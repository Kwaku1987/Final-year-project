"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/pages/mode_selection_page.py

Purpose:
    First screen shown each session: choose Administrator
    (Training Mode) or Lecturer/Advisor (Assessment Mode).
====================================================================
"""

from __future__ import annotations

import streamlit as st

from dashboard.dashboard_controller import DashboardController

from utils.auth import (
    attempts_remaining,
    credential_status,
    is_admin_unlocked,
    reset_attempts,
    verify_admin_password,
)


def render_mode_selection_page(
    controller: DashboardController,
) -> None:

    st.title("🎓 Student Performance Assessment System")

    st.caption(
        "Predict student academic performance (Low / Average / "
        "High) using attendance, study habits, and assessment "
        "scores."
    )

    st.divider()

    st.subheader("Choose how you'd like to use this system")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🛠️ Training Mode")

        st.caption("For Administrators")

        st.write(
            "Upload a dataset, build the target labels, train "
            "and compare models, and save the best one for "
            "lecturers to use."
        )

        status = credential_status()

        if status["kind"] == "default":

            st.warning(
                "🔐 **No administrator password is configured.** The "
                "documented development password (`change-me-admin`) is "
                "active, which means anyone who finds this URL can enter "
                "Training Mode. Set `SPAS_ADMIN_PASSWORD_HASH` in your "
                "host's secrets."
            )

        elif status["kind"] == "plaintext":

            st.info(
                "🔑 A password is configured (plaintext, "
                f"{status['length']} characters). It works, but store the "
                "**hash** instead: "
                "`python -m utils.auth hash \"your-password\"`."
            )

        else:

            st.caption(
                "🔒 Password configured as a SHA-256 hash "
                f"({status['length']} characters)."
            )

        password = st.text_input(
            "Administrator password",
            type="password",
            key="admin_password_input",
            help=(
                "Training Mode can retrain and delete the model "
                "lecturers depend on, so it is password-gated."
            ),
        )

        if attempts_remaining() <= 0:

            st.error(
                "Too many failed attempts in this session — the "
                "password form is paused."
            )

            if st.button(
                "🔁 Reset attempts and try again",
                use_container_width=True,
            ):

                reset_attempts()

                st.rerun()

            st.caption(
                "The attempt counter is a guard against typos, not a "
                "security boundary; the password itself is. Fix the "
                "secret first — expand *Still failing? Diagnose it* "
                "below."
            )

            with st.expander("Still failing? Diagnose it"):

                st.markdown(
                    "Run these in the app's own environment (on "
                    "Community Cloud: **Manage app → Terminal**):\n\n"
                    "```bash\npython -m utils.auth check\n```"
                )

        elif st.button(
            "Enter Training Mode",
            type="primary",
            use_container_width=True,
        ):

            succeeded = verify_admin_password(password)

            controller.log_admin_unlock(succeeded)

            if succeeded:

                controller.set_mode("administrator")

                st.rerun()

            else:

                # A bare "wrong password" hid three different problems:
                # no credential configured, a credential in an
                # unexpected format, and a genuine typo. Say which.
                if attempts_remaining() <= 0:

                    st.error(
                        "Too many failed attempts. Reload the page to "
                        "try again."
                    )

                elif status["kind"] == "default":

                    st.error(
                        "Incorrect password. No password has been "
                        "configured for this deployment, so the "
                        "development password (`change-me-admin`) is in "
                        f"use. {attempts_remaining()} attempt(s) left."
                    )

                else:

                    st.error(
                        "Incorrect password — the value you typed does "
                        "not match the credential configured in "
                        f"`{status['variable']}` ({status['kind']}, "
                        f"{status['length']} characters). "
                        f"{attempts_remaining()} attempt(s) left."
                    )

                with st.expander("Still failing? Diagnose it"):

                    st.markdown(
                        "The credential is read from the host's "
                        "environment or secrets, not from this form.\n\n"
                        "1. **Run the checker on the same host:**\n"
                        "   ```bash\n"
                        "   python -m utils.auth check\n"
                        "   ```\n"
                        "   It prints whether a credential is set, "
                        "whether it is a hash or a plaintext password, "
                        "and never prints the secret itself.\n"
                        "2. **Test a password against it:**\n"
                        "   ```bash\n"
                        "   python -m utils.auth verify \"your-password\"\n"
                        "   ```\n"
                        "3. **Common causes:**\n"
                        "   * the **password** was pasted instead of its "
                        "hash — both are accepted, but the password must "
                        "be the exact one you type here;\n"
                        "   * the secret was saved with the quotes or "
                        "the `SPAS_ADMIN_PASSWORD_HASH =` prefix (both "
                        "are now stripped automatically);\n"
                        "   * the value was changed but the app has not "
                        "rebooted — use *Manage app → Reboot*;\n"
                        "   * the secret is in *Variables* instead of "
                        "*Secrets* — Community Cloud only guarantees "
                        "`st.secrets`; this app bridges those into the "
                        "environment at startup."
                    )

    with col2:

        st.markdown("### 📋 Assessment Mode")

        st.caption("For Lecturers / Academic Advisors")

        st.write(
            "Load the saved model and assess individual "
            "students by entering their details -- no training "
            "required, and no access to the training data."
        )

        model_exists = controller.framework.has_saved_model(
            controller.DEFAULT_MODEL_PATH
        )

        if not model_exists:

            st.warning(
                "No saved model found yet. An Administrator "
                "needs to train and save one first."
            )

        if st.button(
            "Enter Assessment Mode",
            type="primary",
            use_container_width=True,
        ):

            controller.set_mode("lecturer")

            st.rerun()
