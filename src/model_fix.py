import h5py
import json
import pathlib


def print_group_structure(group, indent=0):
    """Helper function to print HDF5 file structure"""
    for name, item in group.items():
        print(" " * indent + name)
        if isinstance(item, h5py.Group):
            print_group_structure(item, indent + 2)
        elif isinstance(item, h5py.Dataset):
            print(" " * indent + f"  → Dataset shape: {item.shape}")
        # Print attributes if they exist
        for attr_name, attr_value in item.attrs.items():
            print(" " * (indent + 2) + f"Attribute - {attr_name}: {attr_value}")


def fix_model_parameters(input_path: str, output_path: str):
    """Fix lr parameter in HDF5 model file"""
    with h5py.File(input_path, "r") as src:
        print("Original file structure:")
        print_group_structure(src)

        # Read the entire file structure
        optimizer_config = None
        for name, item in src.items():
            if isinstance(item, h5py.Group):
                if "optimizer_weights" in name:
                    optimizer_config = item.attrs.get("optimizer_config", None)
                    if optimizer_config is not None:
                        print("\nFound optimizer config:")
                        print(optimizer_config)

        if optimizer_config is None:
            print("\nNo optimizer config found!")
            return

        # Try to decode and modify the config
        try:
            config_dict = json.loads(optimizer_config)
            print("\nDecoded config:")
            print(json.dumps(config_dict, indent=2))

            # Recursively replace 'lr' with 'learning_rate' in the dictionary
            def replace_lr(d):
                if isinstance(d, dict):
                    return {
                        ("learning_rate" if k == "lr" else k): replace_lr(v)
                        for k, v in d.items()
                    }
                elif isinstance(d, list):
                    return [replace_lr(i) for i in d]
                else:
                    return d

            modified_config = replace_lr(config_dict)
            print("\nModified config:")
            print(json.dumps(modified_config, indent=2))

            # Encode back to bytes
            modified_config_bytes = json.dumps(modified_config).encode()

        except json.JSONDecodeError:
            # If JSON parsing fails, try direct byte replacement
            print("\nJSON parsing failed, trying direct byte replacement")
            modified_config_bytes = optimizer_config.replace(
                b'"lr":', b'"learning_rate":'
            )

    # Create new file with modified config
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        # Copy all attributes
        for attr_name, attr_value in src.attrs.items():
            dst.attrs[attr_name] = attr_value

        # Copy all groups and datasets
        for name, item in src.items():
            if isinstance(item, h5py.Group):
                src.copy(name, dst)
                if "optimizer_weights" in name:
                    print(f"\nModifying optimizer config in group: {name}")
                    dst[name].attrs.modify("optimizer_config", modified_config_bytes)
            else:
                src.copy(name, dst, name)

    # Verify the changes
    print("\nVerifying changes in new file:")
    with h5py.File(output_path, "r") as dst:
        print_group_structure(dst)


if __name__ == "__main__":
    data_path = pathlib.Path(__file__).parents[1] / "data"
    fix_model_parameters(
        input_path=data_path / "pretrained_model.hdf5",
        output_path=data_path / "pretrained_model_fixed.hdf5",
    )
