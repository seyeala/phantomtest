# PhantomTest

PhantomTest provides a minimal test harness for verifying connectivity to a variety of data acquisition (DAQ) devices. The repository contains a single script, `test_daqs.py`, that attempts to detect attached hardware from several vendors.

## Driver prerequisites

The test script expects the vendor drivers below to already be installed on the host system:

* **LabJack LJM** – runtime and drivers for LabJack devices
* **MCC UL** – Measurement Computing Universal Library
* **NI-DAQmx Runtime** – runtime libraries for National Instruments DAQ hardware

Install the appropriate driver packages from each vendor before running the test.

## Setup

1. Ensure Python and `pip` are available.
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the test script

Execute the DAQ detection script from the repository root:

```bash
python test_daqs.py
```

The script will report which supported devices are available on the system.

