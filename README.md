# PhantomTest

PhantomTest provides a minimal test harness for verifying connectivity to a variety of data acquisition (DAQ) devices.

## Repository layout

* **Tests**
  * `tests/test_daqs.py` – imports DAQ vendor libraries and reports availability.
  * `tests/test_nidaqmx_devices.py` – enumerates NI‑DAQmx devices.
  * `tests/test_device_names.py` – lists device names and product types.
  * `tests/test_daqo_config.py` – exercises `daqio.daqO.load_config`.
  * `tests/test_publisher_ai.py` – verifies the async publisher writes CSV data.
* **Example scripts**
  * `IOasyncExample.py` – asynchronous analog I/O demo.
  * `IOasyncInteractive.py` – async analog I/O demo with keypress exit.
  * `list_devices.py` – prints detected device names and product types.
  * `test_ai_all.py` – samples all analog‑input channels and averages readings.
  * `test_ao_random.py` – drives analog‑output channels with random voltages.
* **Package**
  * `daqio/` – helpers for NI‑DAQmx I/O and CSV publishing:
    * `config.py` – YAML loading, device discovery and argument parsing.
    * `publisher.py` – async queues for publishing AI/AO data to CSV.
    * `daqI.py` – synchronous analog‑input reader.
    * `daqO.py` – random analog‑output driver.
    * `ai_reader.py` – object‑oriented analog‑input reader with optional publishing.
    * `ao_runner.py` – async analog‑output runner for random values or waveforms.
* **Configs**
  * `configs/` – sample YAML files consumed by the helpers for channel lists and CSV settings.
* **Dependencies**
  * `requirements.txt` – Python dependencies for the scripts.

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

### Configuration utilities (`daqio/config.py`)

`daqio/config.py` centralises helpers for all modules.  It loads YAML files,
discovers NI‑DAQmx devices and merges command‑line options via
`parse_args_with_config`【F:daqio/config.py†L1-L27】【F:daqio/config.py†L107-L152】.
`configs/config_test.yml` shows sample `daqI` and `daqO` sections while
`ai_writer.yml` and `ao_writer.yml` provide minimal CSV writer settings.

### CSV publisher (`daqio/publisher.py`)

`daqio/publisher.py` offers `publish_ai`/`publish_ao` functions that push
measurements through asyncio queues and background consumers that write CSV
rows【F:daqio/publisher.py†L44-L76】【F:daqio/publisher.py†L84-L116】.  Start a
consumer with `start_ai_consumer` or `start_ao_consumer` and supply output
configuration such as `configs/daqI_output.yml` or `configs/daqO_output.yml`.

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

### Object-oriented analog input (`daqio/ai_reader.py`)

`AIReader` wraps NI‑DAQmx input tasks in a context manager and can perform
single reads or buffered `read_average` acquisitions.  Construct it from YAML
with `AIReader.from_yaml("configs/config_test.yml")` and optionally pass
`publish=publish_ai` to stream results formatted by `configs/daqI_output.yml`
【F:daqio/ai_reader.py†L17-L62】【F:daqio/ai_reader.py†L95-L139】.

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

### Async analog output runner (`daqio/ao_runner.py`)

`AsyncAORunner` drives outputs either with random values or by replaying a
waveform.  Use `AsyncAORunner(..., interval=0.5)` for random mode or provide a
`waveform` to play it hardware‑timed at `frequency` cycles per second.  The
runner publishes each update, enabling logging via `configs/daqO_output.yml`
【F:daqio/ao_runner.py†L1-L74】【F:daqio/ao_runner.py†L256-L269】.

