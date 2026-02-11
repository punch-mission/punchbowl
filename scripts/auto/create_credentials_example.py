from prefect.blocks.core import Block
from prefect.blocks.fields import SecretDict
from prefect_sqlalchemy import ConnectionComponents, SqlAlchemyConnector, SyncDriver


class SpacecraftMapping(Block):
    mapping: SecretDict

if __name__ == "__main__":
    connector = SqlAlchemyConnector(
        connection_info=ConnectionComponents(
            driver="mysql+pymysql",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            host="localhost",
            database="punchpipe",
        )
    )

    connector.save("mariadb-creds", overwrite=True)

    mapping_of_ids = {"moc": [1, 2, 3, 4],
                      "soc": [5, 6, 7, 8]}
    mapping = SpacecraftMapping(mapping=mapping_of_ids)
    mapping.save(name="spacecraft-ids", overwrite=True)
