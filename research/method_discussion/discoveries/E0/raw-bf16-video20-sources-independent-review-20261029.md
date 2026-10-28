# Independent review: raw BF16 video20 source cohort

**SUPPORT — fixed development cohort identity.** SHA-256: tool `tools/raw_bf16_video20_sources.py` `0a880233cb62a7930e6fa8c0d177bc926278cfcf902b772e9accf49046f6ce20`; test `tests/test_raw_bf16_video20_sources.py` `5472cf2e01d46c6675e72595e324c18c1b50ed6f9185c2eace2958673a744c4a`; draft `config/e0_raw_bf16_full160_video20_sources_20261029_draft.json` `5548e3191f86db1b5195ed271f0d0e8488e611d80166499f4fee0a5da9599c7e`.

The fixed selection uses 14 consumer sources after excluding the six calibration fit images, three formal subject sources, and three formal camera images with explicitly changed camera prompts. It verifies 20 distinct image hashes and ordinals, excludes the fit and confirmation image clusters, and records 10 subject, 7 background and 3 camera entries, with ten entries per part. All are historically exposed development images; new camera prompts have a distinct conditioning identity. This source draft does not authorize video generation.

The synthetic control `python3 -m unittest discover -s tests -p test_raw_bf16_video20_sources.py` passed 1 test, exit 0. A separate actual image-byte validation checked the 20 image sizes and SHA identities successfully. No model or GPU was used in the review.
