# Local Terrain Estimation


### Install CUDA

Please visit https://developer.nvidia.com/cuda-downloads to download CUDA. Note that if you are Chinese, you can change all the `.com` to `.cn`. More precisely, when you meet the commands like, 
~~~
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
~~~
you need to change them with
~~~
$ wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
~~~

One may meet the following problem when running `sudo apt update` now:
~~~
$ W: GPG error: https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64  InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
$ E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease' is not signed.
$ N: Updating from such a repository can't be done securely, and is therefore disabled by default.
$ N: See apt-secure(8) manpage for repository creation and user configuration details.
~~~
You can run the command below to fix this problem. Please check the version of your system. If you want to find more information, you can refer to this link: https://forums.developer.nvidia.com/t/gpg-error-http-developer-download-nvidia-com-compute-cuda-repos-ubuntu1804-x86-64/212904/3.
~~~
$ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
~~~


<!-- ### Set up the environment
~~~
$ conda create -n local-ter-est python=3.6
$ conda activate local-ter-est
$ pip install pyrealsense2
$ pip install opencv-python
~~~ -->

### Set Up
1. Install virtualenv: 
    ~~~
    $ sudo apt install python3-virtualenv
    ~~~
2. create a virtual environment:
    ~~~
    $ virtualenv --python /usr/bin/python3.8 local-ter-est-env
    ~~~
3. activate the virtual environment:
    ~~~
    $ source ~/Documents/local-terrain-estimation/local-ter-est-env/bin/activate
    ~~~
4. install the dependences:
    ~~~
    $ pip install -r requirements.txt
    ~~~
