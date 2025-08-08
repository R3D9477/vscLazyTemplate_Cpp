### Lazy Template for VSCode (C++)
Template of C++ project adopted for CUDA.<br/>
Full info about `Lazy Template` can be found in [main](https://github.com/R3D9477/vscLazyTemplate_Cpp/blob/main/).

### Before start:
* check cuda version in `invidia-smi`
* then edit `.devcontainer/Dockerfile_ubuntu_cuda`, set appropriate CUDA and Ubuntu version
    * also check `Nsight Compute` version, that going to be installed
* change in `settings.json` option `CppRunCurrentFile.cuda_gpu_architecture` to your own (by default `60` for `Pascal`)
* change in `CMakeLists.txt` line `27` option `CMAKE_CUDA_ARCHITECTURES` to your own (by default `60` for `Pascal`)

### Notes:
* `vcpkg` is supported and enabled
* `Conan2` is not supported and disabled, yet

---

###### If you like that repo, you can support me, I really appreciate it :heart:
[![ko-fi](https://www.ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R3D9477)
