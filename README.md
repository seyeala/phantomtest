# PhantomTest

PhantomTest provides a minimal test harness for verifying connectivity to a variety of data acquisition (DAQ) devices.

## Repository layout

* `test_daqs.py` – imports DAQ vendor libraries (LabJack LJM, MCC UL, NI‑DAQmx) and reports whether each module is available.
* `test_nidaqmx_devices.py` – enumerates NI‑DAQmx devices and prints the name and product type of each detected device.
* `test_device_names.py` – lists device names and product types using the NI‑DAQmx Python API.
* `test_ai_all.py` – reads all analog input channels on a specified NI device, averaging a given number of samples at a specified frequency.
* `test_ao_random.py` – writes random voltages to all analog output channels on a specified NI device.
* `requirements.txt` – Python dependencies needed for the scripts.

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

* Drive random voltages on all analog output channels of a device:

   ```bash
   python test_ao_random.py --dev <DEVICE_NAME> --interval <s> --low <V> --high <V>
   ```

The scripts will report the detected hardware or analog input/output readings.

