import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import dpctl
import _minilb_step


def main():
    print("miniLB step using Python + pybind11 + C++ SYCL")
    print("==============================================")

    dev = dpctl.SyclDevice("opencl:cpu")

    print("Device from dpctl:")
    print(" ", dev.name)

    print("\nDevice name returned by C++:")
    print(" ", _minilb_step.get_device_name(dev))

    print("\nRunning miniLB one-step in C++...")
    result = _minilb_step.run_minilb_step(dev, 8, 8)

    print("\nResults:")
    for key, value in result.items():
        print(f"{key}: {value}")

    if result["check_ok"]:
        print("\nminiLB pybind11 step check: OK")
    else:
        print("\nminiLB pybind11 step check: FAILED")


if __name__ == "__main__":
    main()
