from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
import sqlite3
import uuid
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.models.domain import AccountRole, AuthenticatedUser
from app.repository import SQLiteCaseRepository


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_username(username: str) -> str:
    return username.strip().lower()


def slugify_name(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = "".join(
        character.lower() if character.isalnum() else "-"
        for character in value.strip()
    )
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return collapsed or None


def _hash_password(password: str, salt: bytes, iterations: int) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    ).hex()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("Passwords must be at least 8 characters long.")


class AuthStore(Protocol):
    backend_name: str

    def get_account_by_username(self, username: str) -> dict[str, object] | None:
        ...

    def get_account_by_id(self, user_id: str) -> dict[str, object] | None:
        ...

    def create_account(self, payload: dict[str, object]) -> dict[str, object]:
        ...

    def list_accounts(self) -> list[dict[str, object]]:
        ...

    def store_session(self, payload: dict[str, object]) -> None:
        ...

    def get_session_user(self, token_hash: str) -> dict[str, object] | None:
        ...

    def revoke_session(self, token_hash: str) -> bool:
        ...


class InMemoryAuthStore(AuthStore):
    backend_name = "memory"

    def __init__(self) -> None:
        self._accounts: dict[str, dict[str, object]] = {}
        self._usernames: dict[str, str] = {}
        self._sessions: dict[str, dict[str, object]] = {}

    def get_account_by_username(self, username: str) -> dict[str, object] | None:
        user_id = self._usernames.get(normalize_username(username))
        if user_id is None:
            return None
        account = self._accounts.get(user_id)
        return dict(account) if account is not None else None

    def get_account_by_id(self, user_id: str) -> dict[str, object] | None:
        account = self._accounts.get(user_id)
        return dict(account) if account is not None else None

    def create_account(self, payload: dict[str, object]) -> dict[str, object]:
        username = normalize_username(str(payload["username"]))
        if username in self._usernames:
            raise ValueError(f"Account username already exists: {username}")
        record = dict(payload)
        self._accounts[str(record["user_id"])] = record
        self._usernames[username] = str(record["user_id"])
        return dict(record)

    def list_accounts(self) -> list[dict[str, object]]:
        return [dict(account) for account in self._accounts.values()]

    def store_session(self, payload: dict[str, object]) -> None:
        self._sessions[str(payload["token_hash"])] = dict(payload)

    def get_session_user(self, token_hash: str) -> dict[str, object] | None:
        session = self._sessions.get(token_hash)
        if session is None or session.get("revoked_at"):
            return None
        expires_at = datetime.fromisoformat(str(session["expires_at"]))
        if expires_at <= _utc_now():
            return None
        account = self._accounts.get(str(session["user_id"]))
        if account is None or not account.get("is_active", True):
            return None
        merged = dict(account)
        merged["session_id"] = session["session_id"]
        merged["session_expires_at"] = session["expires_at"]
        return merged

    def revoke_session(self, token_hash: str) -> bool:
        session = self._sessions.get(token_hash)
        if session is None or session.get("revoked_at"):
            return False
        session["revoked_at"] = _utc_now().isoformat()
        return True


class SQLiteAuthStore(AuthStore):
    backend_name = "sqlite"

    def __init__(self, db_path: Path | str, *, timeout_seconds: float = 5.0) -> None:
        self.db_path = db_path
        self.timeout_seconds = timeout_seconds
        if isinstance(db_path, Path):
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS auth_accounts (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    institution_slug TEXT,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_iterations INTEGER NOT NULL,
                    is_active INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_auth_accounts_role_created
                ON auth_accounts(role, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_auth_accounts_institution
                ON auth_accounts(institution_slug)
                WHERE institution_slug IS NOT NULL;

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES auth_accounts(user_id)
                );

                CREATE INDEX IF NOT EXISTS idx_auth_sessions_token_hash
                ON auth_sessions(token_hash);

                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_active
                ON auth_sessions(user_id, expires_at DESC)
                WHERE revoked_at IS NULL;
                """
            )

    def get_account_by_username(self, username: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    user_id,
                    username,
                    display_name,
                    role,
                    institution_slug,
                    password_salt,
                    password_hash,
                    password_iterations,
                    is_active,
                    created_at
                FROM auth_accounts
                WHERE username = ?
                """,
                (normalize_username(username),),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_account_by_id(self, user_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    user_id,
                    username,
                    display_name,
                    role,
                    institution_slug,
                    password_salt,
                    password_hash,
                    password_iterations,
                    is_active,
                    created_at
                FROM auth_accounts
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def create_account(self, payload: dict[str, object]) -> dict[str, object]:
        record = dict(payload)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO auth_accounts (
                        user_id,
                        username,
                        display_name,
                        role,
                        institution_slug,
                        password_salt,
                        password_hash,
                        password_iterations,
                        is_active,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["user_id"],
                        normalize_username(str(record["username"])),
                        record["display_name"],
                        record["role"],
                        record["institution_slug"],
                        record["password_salt"],
                        record["password_hash"],
                        record["password_iterations"],
                        1 if record["is_active"] else 0,
                        record["created_at"],
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Account username already exists: {record['username']}"
            ) from exc
        return record

    def list_accounts(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    user_id,
                    username,
                    display_name,
                    role,
                    institution_slug,
                    is_active,
                    created_at
                FROM auth_accounts
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def store_session(self, payload: dict[str, object]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_sessions (
                    session_id,
                    user_id,
                    token_hash,
                    created_at,
                    expires_at,
                    revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["session_id"],
                    payload["user_id"],
                    payload["token_hash"],
                    payload["created_at"],
                    payload["expires_at"],
                    payload.get("revoked_at"),
                ),
            )

    def get_session_user(self, token_hash: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    account.user_id,
                    account.username,
                    account.display_name,
                    account.role,
                    account.institution_slug,
                    account.is_active,
                    account.created_at,
                    session.session_id,
                    session.expires_at AS session_expires_at
                FROM auth_sessions AS session
                JOIN auth_accounts AS account
                    ON account.user_id = session.user_id
                WHERE session.token_hash = ?
                  AND session.revoked_at IS NULL
                  AND account.is_active = 1
                ORDER BY session.created_at DESC
                LIMIT 1
                """,
                (token_hash,),
            ).fetchone()
        if row is None:
            return None
        if datetime.fromisoformat(str(row["session_expires_at"])) <= _utc_now():
            return None
        return dict(row)

    def revoke_session(self, token_hash: str) -> bool:
        revoked_at = _utc_now().isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE auth_sessions
                SET revoked_at = ?
                WHERE token_hash = ?
                  AND revoked_at IS NULL
                """,
                (revoked_at, token_hash),
            )
            return cursor.rowcount > 0


@dataclass
class TokenIssueResult:
    access_token: str
    expires_at: str
    user: AuthenticatedUser


class AuthService:
    def __init__(self, store: AuthStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self.token_ttl_hours = max(1, settings.auth_token_ttl_hours)
        self.password_iterations = max(100_000, settings.auth_password_iterations)
        self._seed_demo_accounts()

    def _seed_demo_accounts(self) -> None:
        if not self.settings.auth_seed_demo_accounts:
            return
        seed_specs = [
            (
                self.settings.auth_demo_citizen_username,
                self.settings.auth_demo_citizen_password,
                "Citizen Demo",
                AccountRole.citizen,
                None,
            ),
            (
                self.settings.auth_demo_operator_username,
                self.settings.auth_demo_operator_password,
                "Operator Demo",
                AccountRole.operator,
                None,
            ),
            (
                self.settings.auth_demo_reviewer_username,
                self.settings.auth_demo_reviewer_password,
                "Reviewer Demo",
                AccountRole.reviewer,
                None,
            ),
            (
                self.settings.auth_demo_institution_username,
                self.settings.auth_demo_institution_password,
                "Institution Demo",
                AccountRole.institution,
                self.settings.auth_demo_institution_slug,
            ),
            (
                self.settings.auth_demo_admin_username,
                self.settings.auth_demo_admin_password,
                "Admin Demo",
                AccountRole.admin,
                None,
            ),
        ]
        for username, password, display_name, role, institution_slug in seed_specs:
            if self.store.get_account_by_username(username) is not None:
                continue
            self._create_account(
                username=username,
                password=password,
                display_name=display_name,
                role=role,
                institution_slug=institution_slug,
            )

    def _build_user(self, record: dict[str, object]) -> AuthenticatedUser:
        return AuthenticatedUser(
            user_id=str(record["user_id"]),
            username=str(record["username"]),
            display_name=str(record["display_name"]),
            role=AccountRole(str(record["role"])),
            institution_slug=str(record["institution_slug"])
            if record.get("institution_slug")
            else None,
            is_active=bool(record.get("is_active", True)),
            created_at=datetime.fromisoformat(str(record["created_at"]))
            if record.get("created_at")
            else _utc_now(),
        )

    def _create_account(
        self,
        *,
        username: str,
        password: str,
        display_name: str,
        role: AccountRole,
        institution_slug: str | None = None,
    ) -> AuthenticatedUser:
        normalized_username = normalize_username(username)
        if not normalized_username:
            raise ValueError("Username is required.")
        if not display_name.strip():
            raise ValueError("Display name is required.")
        _validate_password(password)
        normalized_institution_slug = slugify_name(institution_slug)
        if role == AccountRole.institution and not normalized_institution_slug:
            raise ValueError("Institution accounts require institution_slug.")
        if role != AccountRole.institution:
            normalized_institution_slug = None
        salt = secrets.token_bytes(16)
        created_at = _utc_now().isoformat()
        record = self.store.create_account(
            {
                "user_id": uuid.uuid4().hex,
                "username": normalized_username,
                "display_name": display_name.strip(),
                "role": role.value,
                "institution_slug": normalized_institution_slug,
                "password_salt": salt.hex(),
                "password_hash": _hash_password(password, salt, self.password_iterations),
                "password_iterations": self.password_iterations,
                "is_active": True,
                "created_at": created_at,
            }
        )
        return self._build_user(record)

    def register_citizen(self, *, username: str, password: str, display_name: str) -> AuthenticatedUser:
        return self._create_account(
            username=username,
            password=password,
            display_name=display_name,
            role=AccountRole.citizen,
        )

    def create_account(
        self,
        *,
        username: str,
        password: str,
        display_name: str,
        role: AccountRole,
        institution_slug: str | None = None,
    ) -> AuthenticatedUser:
        return self._create_account(
            username=username,
            password=password,
            display_name=display_name,
            role=role,
            institution_slug=institution_slug,
        )

    def login(self, *, username: str, password: str) -> TokenIssueResult:
        record = self.store.get_account_by_username(username)
        if record is None or not bool(record.get("is_active", True)):
            raise ValueError("Invalid username or password.")
        salt = bytes.fromhex(str(record["password_salt"]))
        computed_hash = _hash_password(
            password,
            salt,
            int(record["password_iterations"]),
        )
        if not hmac.compare_digest(computed_hash, str(record["password_hash"])):
            raise ValueError("Invalid username or password.")
        access_token = secrets.token_urlsafe(32)
        token_hash = _hash_token(access_token)
        expires_at = (_utc_now() + timedelta(hours=self.token_ttl_hours)).isoformat()
        self.store.store_session(
            {
                "session_id": uuid.uuid4().hex,
                "user_id": record["user_id"],
                "token_hash": token_hash,
                "created_at": _utc_now().isoformat(),
                "expires_at": expires_at,
                "revoked_at": None,
            }
        )
        return TokenIssueResult(
            access_token=access_token,
            expires_at=expires_at,
            user=self._build_user(record),
        )

    def resolve_token(self, token: str) -> AuthenticatedUser | None:
        record = self.store.get_session_user(_hash_token(token))
        if record is None:
            return None
        return self._build_user(record)

    def revoke_token(self, token: str) -> bool:
        return self.store.revoke_session(_hash_token(token))

    def list_accounts(self) -> list[AuthenticatedUser]:
        return [self._build_user(record) for record in self.store.list_accounts()]

    def diagnostics(self) -> dict[str, object]:
        accounts = self.list_accounts()
        counts_by_role: dict[str, int] = {}
        for account in accounts:
            counts_by_role[account.role.value] = counts_by_role.get(account.role.value, 0) + 1
        return {
            "auth_enabled": self.settings.auth_enabled,
            "auth_backend": self.store.backend_name,
            "auth_account_count": len(accounts),
            "auth_accounts_by_role": counts_by_role,
            "auth_token_ttl_hours": self.token_ttl_hours,
        }


def build_auth_service(settings: Settings, repository: object) -> AuthService:
    if isinstance(repository, SQLiteCaseRepository):
        store: AuthStore = SQLiteAuthStore(
            repository.db_path,
            timeout_seconds=repository.timeout_seconds,
        )
    else:
        store = InMemoryAuthStore()
    return AuthService(store=store, settings=settings)
