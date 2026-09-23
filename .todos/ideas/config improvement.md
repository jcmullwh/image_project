config should be generated per experiment. The experiment should be run from the config and it should be persisted as an artifact. There have been several instances where the central config causes conflicts (particularly with full vs prompt only. ie intended to be full but accidentally did prompt only.).

Should "environment" and per-section config be separated? Maybe a utility to automatically generate a skeleton config for a given setup? 



