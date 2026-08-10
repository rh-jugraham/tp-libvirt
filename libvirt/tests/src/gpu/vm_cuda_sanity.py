import os
import re

from virttest import utils_misc
from virttest import virsh

from virttest.libvirt_xml import vm_xml
from virttest.utils_libvirt import libvirt_vmxml
from virttest.utils_libvirt import libvirt_virtio

from provider.gpu import gpu_base


def run(test, params, env):
    """
    cuda sanity tests in guest
    """
    vm_name = params.get("main_vm", "avocado-vt-vm1")
    vm = env.get_vm(vm_name)
    cuda_tests = eval(params.get("cuda_tests", "[]"))
    max_cuda_failures = int(params.get("max_cuda_failures", "0"))
    gpu_test = gpu_base.GPUTest(vm, test, params)
    dev_name = gpu_test.gpu_dev_name
    dev_pci = gpu_test.gpu_pci
    gpu_hostdev_dict = gpu_test.parse_hostdev_dict()
    gpu_managed_disabled = gpu_hostdev_dict.get('managed') != "yes"

    try:
        gpu_test.setup_nvgrace_host_driver()
        gpu_test.setup_default(dev_name=dev_name, test_hopper_gpu="yes")
        
        test.log.info("TEST_STEP: Start the VM")
        vm.start()
        vm_session = vm.wait_for_login(timeout=600)
        test.log.debug(f'VMXML of {vm_name}:\n{virsh.dumpxml(vm_name).stdout_text}')

        test.log.info("TEST_STEP: Clone cuda-samples")
        vm_session.cmd_status_output("rm -rf /root/cuda-samples")
        vm_session.cmd("git clone https://github.com/NVIDIA/cuda-samples --depth 1", timeout=600)

        test.log.info("TEST_STEP: Install the driver")
        vm_session.cmd("dnf clean metadata", timeout=60)
        gpu_test.install_latest_driver(vm_session)

        test.log.info("TEST_STEP: Reboot VM to load new NVIDIA kernel modules")
        vm.reboot(timeout=1200)
        vm_session = vm.wait_for_login(timeout=1200)

        test.log.info("Verifying nvidia modules loaded...")
        s, o = vm_session.cmd_status_output("modprobe nvidia-uvm || true")
        test.log.info(f"modprobe nvidia-uvm: status={s}, output={o}")

        s, o = vm_session.cmd_status_output("nvidia-modprobe -u -c 0 || true")
        test.log.info(f"nvidia-modprobe: status={s}, output={o}")

        s, o = vm_session.cmd_status_output("modprobe nvidia")
        test.log.info(f"modprobe nvidia: status={s}, output={o}")

        s, o = vm_session.cmd_status_output("ls -l /dev/nvidia* || true")
        test.log.info(f"/dev/nvidia* devices: {o}")

        s, o = vm_session.cmd_status_output("lsmod | grep nvidia")
        test.log.info(f"nvidia modules after modprobe: {o}")

        s, o = vm_session.cmd_status_output("dmesg | tail -50")
        test.log.info(f"dmesg after modprobe: {o}")

        s, o = vm_session.cmd_status_output("systemctl start nvidia-persistenced || true")
        test.log.info(f"start nvidia-persistenced: status={s}, output={o}")

        # Wait for NUMA nodes to initialize
        vm_session.cmd("sleep 30")
        test.log.info("Waited 30s for nvidia-persistenced to initialize NUMA nodes")

        s, o = vm_session.cmd_status_output("dmesg | grep -i 'NUMA'")
        test.log.info(f'dmesg numa {o}')

        s, o = vm_session.cmd_status_output("dmesg | grep -i 'NUMA was not set up'")
        if o:
            test.log.warn(f"NUMA setup warning found: {o}")

        s, o = vm_session.cmd_status_output("numactl --hardware")
        test.log.info(f"NUMA nodes: {o}")

        gpu_test.install_cuda_toolkit(vm_session)
        gpu_test.nvidia_smi_check(vm_session)

        test.log.info("TEST_STEP: Build CUDA samples")
        vm_session.cmd("dnf install -y git cmake mesa-libGL-devel freeglut-devel vulkan-headers vulkan-loader-devel mesa-vulkan-drivers glfw-devel gcc-c++", timeout=600)        
        vm_session.cmd(f"rm -rf /root/cuda-samples/build")
        vm_session.cmd(f"mkdir /root/cuda-samples/build")
        vm_session.cmd(f"sed -i 's/add_subdirectory(9_CUDA_Tile)/#add_subdirectory(9_CUDA_Tile)/' /root/cuda-samples/cpp/CMakeLists.txt || true")
        vm_session.cmd(f"sed -i 's/add_subdirectory(UnifiedMemoryStreams)/#add_subdirectory(UnifiedMemoryStreams)/' /root/cuda-samples/Samples/0_Introduction/CMakeLists.txt || true")
        vm_session.cmd(f"cd /root/cuda-samples/build && export PATH=$PATH:/usr/local/cuda/bin/ && cmake -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON ..", timeout=600)
        vm_session.cmd(f"cd /root/cuda-samples/build && make -j 8", timeout=1200)

        test.log.info("TEST_STEP: Run cuda sanity tests")
        if cuda_tests:
            # select_tests
            for s_test in cuda_tests:
                test.log.debug(f"TEST_STEP: Run {s_test}")
                s, o = vm_session.cmd_status_output(f"LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH {s_test}", timeout=600)    
                test.log.debug(f"output: {o}")        
                if s:
                    test.fail("Failed to run cuda sanity - %s" % s_test)

        else:
            # all_tests
            test.log.info("TEST_STEP: Run cuda sanity tests using run_tests.py")
            s, o = vm_session.cmd_status_output(f"cd /root/cuda-samples && export PATH=$PATH:/usr/local/cuda/bin/ && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && python3 run_tests.py --dir ./build/cpp --output results", timeout=2000)        
            test.log.info(f"run_tests.py output: {o}")
            
            # Check if test summary exists
            if "Test Summary:" not in o:
                test.fail("run_tests.py did not complete properly - no Test Summary found")

            # Parse failure count from output
            failed_match = re.search(r'Failed runs \((\d+)\)', o)
            failed_count = int(failed_match.group(1)) if failed_match else 0

            # Pass/fail based on threshold
            if failed_count > max_cuda_failures:
                test.fail(f"CUDA tests FAILED: {failed_count} failures exceeds threshold of {max_cuda_failures}")
            else:
                test.log.info(f"CUDA tests PASSED: {failed_count} failures within acceptable threshold of {max_cuda_failures}")

    finally:
        gpu_test.teardown_default(
            managed_disabled=gpu_managed_disabled,
            dev_name=dev_name
        )
