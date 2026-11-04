"""Synthetic smoke checks for account registration and authentication."""
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from auth import (
    UsernameExistsError,
    authenticate_user,
    is_authenticated,
    logout,
    register_user,
    validate_registration,
)
from streamlit.testing.v1 import AppTest


def update_input(app, key, value):
    app.text_input(key=key).set_value(value)


def click_button(app, key):
    app.button(key=key).click().run()
    assert not app.exception, [str(item.value) for item in app.exception]


def check_dashboard_auth_flow(database_path):
    os.environ["ATTENDANCE_AUTH_DB_PATH"] = str(database_path)
    dashboard = Path(__file__).with_name("dashboard.py")
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    assert not app.exception, [str(item.value) for item in app.exception]
    assert app.button(key="show_signup")
    assert not app.sidebar.radio
    assert app.text_input(key="login_password").proto.type == 1

    click_button(app, "show_signup")
    click_button(app, "FormSubmitter:signup_form-Create Account")
    assert "Username cannot be empty." in app.error[0].value
    assert not app.sidebar.radio

    update_input(app, "signup_username", "teacher.one")
    click_button(app, "FormSubmitter:signup_form-Create Account")
    assert "Password cannot be empty." in app.error[0].value

    update_input(app, "signup_password", "correct horse")
    update_input(app, "signup_confirmation", "different")
    click_button(app, "FormSubmitter:signup_form-Create Account")
    assert "Passwords do not match." in app.error[0].value

    update_input(app, "signup_confirmation", "correct horse")
    click_button(app, "FormSubmitter:signup_form-Create Account")
    assert "Account created." in app.success[0].value
    click_button(app, "show_signup")
    update_input(app, "signup_username", "TEACHER.ONE")
    update_input(app, "signup_password", "another password")
    update_input(app, "signup_confirmation", "another password")
    click_button(app, "FormSubmitter:signup_form-Create Account")
    assert "already registered" in app.error[0].value

    click_button(app, "show_login")
    update_input(app, "login_username", "teacher.one")
    update_input(app, "login_password", "wrong password")
    click_button(app, "FormSubmitter:login_form-Login")
    assert "username or password is incorrect" in app.error[0].value
    update_input(app, "login_username", "teacher.one")
    update_input(app, "login_password", "correct horse")
    click_button(app, "FormSubmitter:login_form-Login")
    assert app.session_state["authenticated_user"] == "teacher.one"
    assert app.button(key="logout")
    assert app.sidebar.radio

    click_button(app, "logout")
    assert "authenticated_user" not in app.session_state
    assert app.button(key="show_signup")
    assert not app.sidebar.radio


def main():
    with tempfile.TemporaryDirectory(prefix="attendance-auth-smoke-") as temporary:
        database = Path(temporary) / "auth.sqlite3"

        assert validate_registration("  teacher.one  ", "secret", "secret") == "teacher.one"
        for username, password, confirmation, expected in (
            ("   ", "secret", "secret", "Username cannot be empty."),
            ("teacher.one", "", "", "Password cannot be empty."),
            ("teacher.one", "secret", "different", "Passwords do not match."),
        ):
            try:
                validate_registration(username, password, confirmation)
            except ValueError as error:
                assert str(error) == expected
            else:
                raise AssertionError(f"Expected validation failure: {expected}")

        register_user("Teacher.One", "correct horse", database)
        assert authenticate_user("teacher.one", "correct horse", database)
        assert not authenticate_user("TEACHER.ONE", "wrong password", database)
        assert not authenticate_user("unknown", "correct horse", database)

        try:
            register_user("teacher.one", "another password", database)
        except UsernameExistsError:
            pass
        else:
            raise AssertionError("Case-insensitive duplicate username was accepted")

        with closing(sqlite3.connect(database)) as connection:
            row = connection.execute(
                "SELECT password_salt, password_hash FROM auth_users"
            ).fetchone()
        assert len(row[0]) == 16
        assert len(row[1]) == 32
        assert b"correct horse" not in row[1]

        session_state = {}
        assert not is_authenticated(session_state)
        session_state["authenticated_user"] = "Teacher.One"
        assert is_authenticated(session_state)
        logout(session_state)
        assert not is_authenticated(session_state)
        check_dashboard_auth_flow(Path(temporary) / "dashboard-auth.sqlite3")
        del os.environ["ATTENDANCE_AUTH_DB_PATH"]
        print("PASS: registration, validation, duplicate rejection, login, logout, session gate, and hashing")


if __name__ == "__main__":
    main()
