Configuration
==============

How to set up a configuration file.

Initializing Prefect
---------------------

If you're processing large amounts of data, it's preferred to use
postgres as the database backend instead of sqlite, Prefect's default.
This allows for more throughput and less database locking. To change the
database backend:

1. Ensure you have postgres installed and accessible
2. After replacing username, password, and database name, run:
`prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://username:password@localhost:5432/db_name"`
3. Now you're switched over to a postgres backend
