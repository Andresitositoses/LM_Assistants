from pathlib import Path

from components.central_unit import CentralUnit, ConfigLoadError


def main():
    config_path = Path(__file__).resolve().parent / "config.ini"
    try:
        central_unit = CentralUnit.from_file(config_path)
    except ConfigLoadError as exc:
        raise SystemExit(f"Error de configuración: {exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    central_unit.run()


if __name__ == "__main__":
    main()
