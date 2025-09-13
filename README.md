# PhantomTest

PhantomTest provides a minimal test harness for verifying connectivity to a variety of data acquisition (DAQ) devices.

## Repository layout

* `tests/test_daqs.py` – imports DAQ vendor libraries (LabJack LJM, MCC UL, NI‑DAQmx) and reports whether each module is available.
* `tests/test_nidaqmx_devices.py` – enumerates NI‑DAQmx devices and prints the name and product type of each detected device.
* `tests/test_device_names.py` – lists device names and product types using the NI‑DAQmx Python API.
* `test_ai_all.py` – reads all analog input channels on a specified NI device, averaging a given number of samples at a specified frequency.
* `test_ao_random.py` – drives analog-output channels with random voltages within a user-specified range.
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
   pytest tests/test_daqs.py
   ```

* List NI‑DAQmx devices:

   ```bash
   pytest tests/test_nidaqmx_devices.py
   ```

* Print device names and types:

   ```bash
   pytest tests/test_device_names.py
   ```

* List device names and product types with a helper script:

   ```bash
   python list_devices.py
   ```

* Sample analog inputs from all channels on a device:

   ```bash
   python test_ai_all.py --dev <DEVICE_NAME> --freq <Hz> --n <samples>
   ```

* Drive analog outputs with random voltages:

   ```bash
   python test_ao_random.py --dev <DEVICE_NAME> --interval <s> [--low <V>] [--high <V>]
   ```

The scripts will report the detected hardware, analog input readings or output activity.

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

Configurations are split into `daqI` (analog input) and `daqO` (analog output)
sections. `daqI` blocks are **only** for inputs; `daqO` blocks are the sole
safe source for outputs. Mixing these sections can raise NI‑DAQmx `I/O type`
errors or drive unintended channels. `IOasyncExample.py` demonstrates loading
each section separately when performing simultaneous input and output.

### Analog input (`daqio/daqI.py`)

`daqio/daqI.py` reads one or more analog-input channels and prints the average
voltage per channel. Configuration is supplied via YAML and must define the
device name, channel list, sample frequency, number of averages, and the number
of omitted intervals between reads. Omitting the `omissions` field raises a
configuration error. The terminal configuration is optional【F:daqio/daqI.py†L1-L26】【F:daqio/daqI.py†L66-L74】.
Use only the `daqI` section for these settings; sourcing channel lists from
`daqO` may cause NI‑DAQmx `I/O type` errors or unintended output.

Example configuration:

```yaml
daqI:
  device: Dev1
  channels:
    - Dev1/ai0
    - Dev1/ai1
  freq: 10
  averages: 5
  omissions: 0  # number of sample intervals to skip between reads
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
seed【F:daqio/daqO.py†L1-L16】【F:daqio/daqO.py†L39-L55】.
Always source these values from the `daqO` section; using `daqI` data for outputs
can raise NI‑DAQmx `I/O type` errors or drive unintended channels.

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

