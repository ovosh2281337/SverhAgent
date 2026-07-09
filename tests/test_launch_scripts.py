import shutil
import subprocess
import unittest
from pathlib import Path


class PowerShellLauncherSmokeTests(unittest.TestCase):
    def test_powershell_scripts_are_ascii_only(self) -> None:
        for path in ("scripts/setup.ps1", "scripts/run.ps1"):
            data = Path(path).read_bytes()
            self.assertTrue(
                all(byte < 128 for byte in data),
                msg=f"{path} contains non-ASCII bytes; keep PS 5.1 launchers ASCII-only",
            )

    @unittest.skipIf(shutil.which("powershell") is None, "Windows PowerShell not available")
    def test_setup_and_run_parse_in_windows_powershell(self) -> None:
        command = (
            "$ErrorActionPreference='Stop';"
            "[scriptblock]::Create((Get-Content -Raw 'scripts/setup.ps1')) | Out-Null;"
            "[scriptblock]::Create((Get-Content -Raw 'scripts/run.ps1')) | Out-Null"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
