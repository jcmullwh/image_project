import subprocess
import sys


def test_generate_module_does_not_emit_pydub_ffmpeg_warning():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import warnings; warnings.simplefilter('default'); "
                "import image_project.app.generate"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "Couldn't find ffmpeg or avconv" not in combined
