#!/usr/bin/env bash
# Create the app role, the database (if it doesn't exist yet) and the pgvector extension on the
# local Postgres, then write the connection settings to .env. Safe to run again.
#
# The app role's password is generated here and only ever stored in .env.
# psql asks for the postgres superuser password (the one you chose in the installer).
#
# Usage: bash scripts/init_db.sh [db_name]     (default: PG_DB from .env, else promo_rag_dev)
# Run it again with another name (e.g. rag_eval) to add a database for the same role.
set -euo pipefail

cd "$(dirname "$0")/.."
ROLE="promo_rag"
PSQL="${PSQL:-/Library/PostgreSQL/18/bin/psql}"

touch .env
env_get() { sed -n "s/^$1=//p" .env | tail -1; }
env_set() {  # replace the key's line if present, otherwise append it
  if grep -q "^$1=" .env; then sed -i '' "s|^$1=.*|$1=$2|" .env; else printf '%s=%s\n' "$1" "$2" >> .env; fi
}

DB="${1:-$(env_get PG_DB)}"; DB="${DB:-promo_rag_dev}"
PORT="$(env_get PG_PORT)"; PORT="${PORT:-5432}"
PW="$(env_get PG_PASSWORD)"; PW="${PW:-$(openssl rand -hex 16)}"

echo "Setting up role '$ROLE' and database '$DB' on localhost:$PORT"
# The password goes in through stdin (\set), not the command line, so it never shows in `ps`.
"$PSQL" -X -q -t -A -h localhost -p "$PORT" -U postgres -d postgres -v ON_ERROR_STOP=1 \
  -v role="$ROLE" -v db="$DB" <<SQL
\set pw $PW
SELECT format('CREATE ROLE %I LOGIN', :'role')
 WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'role') \gexec
SELECT format('ALTER ROLE %I PASSWORD %L', :'role', :'pw') \gexec
SELECT format('CREATE DATABASE %I OWNER %I', :'db', :'role')
 WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'db') \gexec
\c :db
SELECT '  owner: ' || pg_get_userbyid(datdba) FROM pg_database WHERE datname = current_database();
SELECT '  tables already here: ' || coalesce(string_agg(schemaname || '.' || tablename, ', '), 'none')
  FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema');
SELECT '  extensions already here: ' || string_agg(extname || ' ' || extversion, ', ') FROM pg_extension;
CREATE EXTENSION IF NOT EXISTS vector;
-- If the database belongs to another role, the app role still needs to create tables.
GRANT USAGE, CREATE ON SCHEMA public TO :"role";
SELECT '  pgvector ' || extversion || ' is enabled in ' || current_database()
  FROM pg_extension WHERE extname = 'vector';
SQL

env_set PG_PASSWORD "$PW"
env_set PG_USER "$ROLE"
[ -n "$(env_get PG_HOST)" ] || env_set PG_HOST localhost
[ -n "$(env_get PG_PORT)" ] || env_set PG_PORT "$PORT"
[ -n "$(env_get PG_DB)" ]   || env_set PG_DB "$DB"
echo "Done. Connection settings are in .env."
