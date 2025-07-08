import toml
import ruamel.yaml

PYPROJECT_FILE = "pyproject.toml"
CITATION_FILE = "CITATION.cff"

def main():
    # Load pyproject.toml
    pyproject = toml.load(PYPROJECT_FILE)
    version = pyproject.get("project", {}).get("version")

    if version is None:
        print("Version not found in pyproject.toml!")
        return

    # Load CITATION.cff
    yaml = ruamel.yaml.YAML()
    with open(CITATION_FILE, "r") as f:
        data = yaml.load(f)

    # Update version
    old_version = data.get("version")
    data["version"] = version

    if old_version != version:
        print(f"Updated CITATION.cff version: {old_version} → {version}")
    else:
        print(f"CITATION.cff already up-to-date (version {version})")

    # Write back
    with open(CITATION_FILE, "w") as f:
        yaml.dump(data, f)

if __name__ == "__main__":
    main()
