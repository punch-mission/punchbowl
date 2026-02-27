import os

import click
import pandas as pd

from punchbowl.auto.level0.ccsds import PACKET_NAME2APID


def _prepare_packet_definition(contents):
    definition = []
    for row in contents.iterrows():
        name = row[1].iloc[0]
        kind = 'uint' if name not in ("FILL_VALUE", "FSW_MEM_DUMP_DATA") else "fill"
        size = row[1].iloc[8]
        definition.append(dict(name=name, data_type=kind, bit_length=size))
    return definition[6:]


def convert_eng_packet(tlm_path, output_dir, packet_name):
    contents = pd.read_excel(tlm_path, sheet_name=packet_name)
    definition = _prepare_packet_definition(contents)
    pd.DataFrame(definition).to_csv(os.path.join(output_dir, f"{packet_name}.csv"), index=False)


def convert_sci_xfi_packet(tlm_path, output_dir):
    contents = pd.read_excel(tlm_path, sheet_name="SCI_XFI")
    definition = _prepare_packet_definition(contents)
    definition[-1]['data_type'] = 'uint(expand)'
    definition[-1]['bit_length'] = 8
    pd.DataFrame(definition).to_csv(os.path.join(output_dir, f"SCI_XFI.csv"), index=False)


@click.command()
@click.argument("tlm_path")
@click.argument("output_dir")
def convert_tlm_to_csv(tlm_path: str, output_dir: str) -> None:
    for packet_name in PACKET_NAME2APID:
        if packet_name != "SCI_XFI":
            convert_eng_packet(tlm_path, output_dir, packet_name)
        else:
            convert_sci_xfi_packet(tlm_path, output_dir)


if __name__ == "__main__":
    convert_tlm_to_csv()
