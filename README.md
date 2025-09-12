# PhantomTest

PhantomTest provides a minimal test harness for verifying connectivity to a variety of data acquisition (DAQ) devices.

## Repository layout

* `test_daqs.py` – imports DAQ vendor libraries (LabJack LJM, MCC UL, NI‑DAQmx) and reports whether each module is available.
* `test_nidaqmx_devices.py` – enumerates NI‑DAQmx devices and prints the name and product type of each detected device.
* `test_device_names.py` – lists device names and product types using the NI‑DAQmx Python API.
* `test_ai_all.py` – reads all analog input channels on a specified NI device, averaging a given number of samples at a specified frequency.
* `requirements.txt` – Python dependencies needed for the scripts.
* `configs/` – sample YAML configuration files.

## Driver prerequisites

The scripts expect the vendor drivers below to already be installed on the host system:

* **LabJack LJM** – runtime and drivers for LabJack devices
* **MCC UL** – Measurement Computing Universal Library
* **NI‑DAQmx Runtime** – runtime libraries for National Instruments DAQ hardware

Install the appropriate driver packages from each vendor before running the scripts.

## Setup

1. Ensure Python and `pip` are available.
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the scripts

Execute the scripts from the repository root:

* Detect available drivers:

   ```bash
   python test_daqs.py
   ```

* List NI‑DAQmx devices:

   ```bash
   python test_nidaqmx_devices.py
   ```

* Print device names and types:

   ```bash
   python test_device_names.py
   ```

* Sample analog inputs from all channels on a device:

   ```bash
   python test_ai_all.py --dev <DEVICE_NAME> --freq <Hz> --n <samples>
   ```

The scripts will report the detected hardware or analog input readings.

## Running tests

Install the test dependencies:

```bash
pip install -r requirements-dev.txt
```

Then run the test suite:

```bash
pytest
```

## DAQ I/O helpers

The `daqio` package provides lightweight helpers for working with NI-DAQmx
analog channels. It requires the NI‑DAQmx runtime and the Python `nidaqmx`
package to be installed. Sample configuration files live under `configs/`.

### Analog input (`daqio/daqI.py`)

`daqio/daqI.py` reads one or more analog-input channels and prints the average
voltage per channel. Configuration is supplied via YAML and must define the
device name, channel list, sample frequency and number of averages; the
terminal configuration is optional【F:daqio/daqI.py†L1-L22】【F:daqio/daqI.py†L55-L76】.

Example configuration:

```yaml
daqI:
  device: Dev1
  channels:
    - Dev1/ai0
    - Dev1/ai1
  freq: 10
  averages: 5
  terminal: RSE
```

Run the module as a script:

```bash
python -m daqio.daqI --config configs/config_test.yml
```

See the module docstring for details on the expected schema and additional
options.

### Analog output (`daqio/daqO.py`)

`daqio/daqO.py` continuously drives analog-output channels with random voltages
generated within a user-specified range. Its YAML configuration accepts the
device name, optional channel list, update interval, voltage bounds and random
seed【F:daqio/daqO.py†L1-L13】【F:daqio/daqO.py†L33-L48】.

Example configuration:

```yaml
daqO:
  device: Dev1
  channels:
    - Dev1/ao0
    - Dev1/ao1
  interval: 0.5
  low: 0.0
  high: 3.0
  seed: 1234
```

Run the module as a script:

```bash
python -m daqio.daqO --config configs/config_test.yml
```

**Safety note:** the module resets all outputs to `0 V` on exit, even when
interrupted with `Ctrl+C`, to avoid leaving channels in an unsafe state【F:daqio/daqO.py†L10-L13】【F:daqio/daqO.py†L143-L147】.

Consult the docstring for further information about the CLI and configuration
options.

