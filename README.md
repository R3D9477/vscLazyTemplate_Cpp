### Lazy Template for VSCode (C++)
Template of C++ project adopted for CUDA.<br/>
Full info about `Lazy Template` can be found in [main](https://github.com/R3D9477/vscLazyTemplate_Cpp/blob/main/).

### Note
* before start check cuda version in `invidia-smi`
* then edit `.devcontainer/Dockerfile_ubuntu_cuda` line `1` and set appropriate CUDA and Ubuntu version
* to install `Nsight Compute` edit line `23` and set appropriate version of Ubuntu repository
    * keep in mind that `Pascal` architecture has been dropped after version `2019.5.1`, so you have to install it manually
* `vcpkg` is supported
* `Conan2` is not supported, yet

---

###### If you like that repo, you can support me, I really appreciate it :heart:
[![ko-fi](https://www.ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R3D9477)
