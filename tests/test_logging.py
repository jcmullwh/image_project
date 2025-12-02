import os
import tempfile
import unittest

from main import write_messages_log


class WriteMessagesLogTests(unittest.TestCase):
    def test_writes_unicode_text_with_utf8_encoding(self):
        unicode_text = "Log entry with arrow \\u2192 and accents \\u00e9."
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "log.txt")

            write_messages_log(log_path, unicode_text)

            with open(log_path, "r", encoding="utf-8") as file:
                content = file.read()

        self.assertEqual(content, unicode_text)


if __name__ == "__main__":
    unittest.main()
