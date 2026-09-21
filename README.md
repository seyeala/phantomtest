# PhantomTest

**PhantomTest** is a Python test and experimentation framework for data
acquisition (DAQ) systems. It provides reusable utilities, executable examples,
and automated tests for discovering DAQ hardware, acquiring analog signals,
generating analog outputs, replaying waveforms, and recording measurements.

## Abstract

Reproducible experimental workflows require both reliable communication with
instrumentation and an explicit record of acquisition parameters. PhantomTest
supports the development and validation of such workflows across several DAQ
platforms, with its primary I/O abstractions built around NI-DAQmx. The
repository combines synchronous and asynchronous analog I/O, YAML-based
configuration, CSV publishing, device-discovery utilities, and hardware-aware
tests. It is intended as a compact foundation for laboratory prototyping,
integration testing, and repeatable data-acquisition experiments.

## Key capabilities

- Discover installed DAQ drivers and enumerate NI-DAQmx devices.
- Acquire single or averaged measurements from one or more analog-input
  channels.
- Generate bounded random analog-output values or replay sampled waveforms.
- Coordinate analog input and output through `asyncio`.
- Publish timestamped AI and AO measurements to CSV files.
- Create standardized capture filenames and coordinate waveform output with
  Pico-based capture workflows.
- Validate configuration and I/O behavior with hardware-independent and
  hardware-dependent tests.

## System requirements

### Software

- Python 3.8 or later
- `pip` and a Python virtual environment (recommended)
- The Python packages declared in `pyproject.toml` or `requirements.txt`

The core `daqio` package depends on
[`nidaqmx`](https://pypi.org/project/nidaqmx/) and
[`PyYAML`](https://pypi.org/project/PyYAML/). The broader driver-detection
examples also use `labjack-ljm` and `mcculw`.

### Vendor drivers and hardware

Install the native runtime appropriate to the hardware that will be used:

- **NI-DAQmx Runtime** for National Instruments hardware
- **LabJack LJM** for LabJack hardware
- **MCC Universal Library** for Measurement Computing hardware

Python packages do not replace these vendor runtimes. Tests that communicate
with physical devices require compatible hardware, drivers, and device names;
the remaining unit tests can be run without attached DAQ hardware.

## Installation

Clone the repository and enter its directory:

```bash
git clone <repository-url>
cd PhantomTest
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the package in editable mode, including its test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

To exercise driver detection for all supported vendors, install the additional
bindings listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run all commands from the repository root. Examples below use placeholder
device names; replace them with the identifiers reported by the local DAQ
runtime.

### Detect drivers and devices

Check whether the supported vendor libraries can be imported:

```bash
python -m pytest tests/test_daqs.py
```

List connected NI-DAQmx devices and their product types:

```bash
list-devices
```

The equivalent source-tree command is `python list_devices.py`.

### Acquire analog input

Sample every analog-input channel on a device and report averaged values:

```bash
python test_ai_all.py --dev Dev1 --freq 1000 --n 10
```

Alternatively, run the configurable input module:

```bash
python -m daqio.daqI --config configs/config_test.yml
```

The `daqI` configuration must define a device, input channels, sampling
frequency, number of samples to average, and the number of intervals omitted
between reads. The terminal configuration is optional.

### Generate analog output

Drive selected output channels with reproducible random values:

```bash
python test_ao_random.py \
  --dev cDAQ1Mod1 \
  --channels cDAQ1Mod1/ao0 cDAQ1Mod1/ao1 \
  --interval 0.5 \
  --low 0.0 \
  --high 3.0 \
  --seed 1234
```

The YAML-configured equivalent is:

```bash
python -m daqio.daqO --config configs/config_test.yml
```

> **Safety:** Confirm channel assignments, wiring, and voltage limits before
> enabling analog output. The `daqO` command resets configured outputs to
> `0 V` when it exits, including after a keyboard interrupt.

### Run asynchronous I/O

`IOasyncExample.py` concurrently performs analog input and output and publishes
both streams to CSV files:

```bash
python IOasyncExample.py
```

`IOasyncInteractive.py` provides an interactive variant with a keypress-based
exit. Both examples read their default settings from `configs/config_test.yml`.

### Coordinate waveform output and capture

The automation entry point under `scripts/` coordinates waveform generation,
AI/AO logging, and Pico capture:

```bash
python scripts/run_waveform_and_pico_capture.py --help
```

Use the help output to select the capture implementation, configuration,
waveform, output directory, duration, and naming options appropriate to the
experiment.

## Configuration

Configuration files are written in YAML. Analog input and output settings must
remain in distinct top-level sections:

```yaml
daqI:
  device: Dev1
  channels:
    - Dev1/ai0
    - Dev1/ai1
  freq: 10000
  averages: 10
  omissions: 20
  terminal: RSE

daqO:
  device: cDAQ1Mod1
  channels:
    - cDAQ1Mod1/ao0
    - cDAQ1Mod1/ao1
  interval: 0.5
  low: 0.0
  high: 3.0
  seed: 1234
```

The `daqI` section is exclusively for input channels, and the `daqO` section is
exclusively for output channels. Interchanging them can produce NI-DAQmx I/O
type errors or drive an unintended channel.

Available examples include:

- `configs/config_test.yml`: combined AI and AO settings
- `configs/ai_writer.yml` and `configs/ao_writer.yml`: minimal CSV publisher
  settings
- `configs/daqI_output.yml` and `configs/daqO_output.yml`: output schemas for
  timestamped measurements
- `configs/OstreamTimeformat.yml`: output-stream time formatting

## Repository structure

| Path | Purpose |
| --- | --- |
| `daqio/` | Configuration, NI-DAQmx I/O, asynchronous runners, and CSV publishing |
| `configs/` | Example acquisition and output configurations |
| `scripts/` | Higher-level waveform and capture automation |
| `tests/` | Unit, integration, and hardware-discovery tests |
| `IOasyncExample.py` | Concurrent AI/AO demonstration |
| `IOasyncInteractive.py` | Interactive concurrent AI/AO demonstration |
| `list_devices.py` | NI-DAQmx device enumeration |
| `test_ai_all.py` | Command-line analog-input sampler |
| `test_ao_random.py` | Command-line random analog-output generator |
| `capture_filename.py` | Standardized capture-name utility |

## Testing

Install the development dependencies if they were not installed through the
`test` extra:

```bash
python -m pip install -r requirements-dev.txt
```

Run the complete test suite with:

```bash
python -m pytest
```

Individual hardware checks can be selected explicitly, for example:

```bash
python -m pytest tests/test_nidaqmx_devices.py
python -m pytest tests/test_device_names.py
```

Hardware-dependent checks require the corresponding runtime and attached
device. Review the target channels and voltage constraints before executing
any output test.

## Citation

If PhantomTest contributes to published work, cite the specific release or
commit used so that the software configuration is reproducible. Until a DOI or
`CITATION.cff` record is available, the following BibTeX template may be
adapted with the repository URL, year, and access date:

```bibtex
@software{phantomtest,
  author  = {{PhantomTest contributors}},
  title   = {PhantomTest: Data-Acquisition Testing and Experimentation Utilities},
  version = {0.1.0},
  url     = {<repository-url>},
  year    = {<release-year>},
  note    = {Accessed: <YYYY-MM-DD>}
}
```

## Funding and acknowledgments

Research reported in this publication was supported by the National Institute
of Biomedical Imaging and Bioengineering of the National Institutes of Health
under Award Number R21EB030654. The content is solely the responsibility of the
authors and does not necessarily represent the official views of the National
Institutes of Health.
