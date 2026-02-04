import sys
import os
import time
import logging
import yaml

from camera_manager import CameraManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("v5_system")


def main() -> None:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if len(sys.argv) >= 2:
        config_path = str(sys.argv[1])
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    try:
        mgr = CameraManager(cfg)
    except Exception as e:
        logger.error(f"Failed to start CameraManager: {e}")
        return

    mgr.start()
    try:
        last_log = time.time()
        while True:
            time.sleep(1.0)
            if time.time() - last_log >= 5.0:
                counts, avg_kmh = mgr.snapshot()
                logger.info(f"Aggregated samples per cam: {counts} | avg_speed={avg_kmh:.1f} km/h")
                last_log = time.time()
            if mgr.all_stopped():
                break
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()
        logger.info("System stopped.")


if __name__ == "__main__":
    main()
