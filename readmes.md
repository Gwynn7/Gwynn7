# 5. 一体机开发
## 5.1 开发板介绍
## 5.2 平台仿真
### 5.2.1 环境配置
#### 5.2.1.1 资料下载
资料来源：
下载站台：sftp://218.17.249.213
账号：cvitek_mlir_2023
密码：7&2Wd%cu5k
Windows系统下载：
使用Windows自带的FTP工具。
首先进入此电脑，空白处点击右键，选择“添加一个网络位置”：


输入ftp://218.17.249.213

输入用户名：

网络名称可以自定义：

创建完成之后会进入页面：

进入文件夹即可看到资料，若要下载只需将其复制进入自己的本地文件夹即可：

需要下载的包：


根据使用EVB板选择tpu_sdk，我使用的EVB板是cv1811c_wevb_0006a,使用的C库是musl,所以选择：

Ubuntu系统下载：
使用ftp之前先关闭防火墙：
iptables -F
使用Ubuntu的FTP工具登录：
orla@ubuntu:~/workspace/cvitek$ ftp
ftp> open 218.17.249.213
Connected to 218.17.249.213.
220 SZ_NAS FTP server ready.
Name (218.17.249.213:orla): cvitek_mlir_2023
331 Password required for cvitek_mlir_2023.
Password: 7&2Wd%cu5k
230 User cvitek_mlir_2023 logged in.
Remote system type is UNIX.
Using binary mode to transfer files.
使用get命令下载文件：
ftp> get cvitek_mlir_ubuntu-18.04_v1.5.0-883-gee0cbe9e3.tar.gz
若要一次下载多个文件，使用mget命令：
#不使用prompt的话，每一个文件都需要询问后才会开始下载。
ftp> prompt
Interactive mode off.
ftp> mget xxx,xxx,xxx
#### 5.2.1.2 环境安装
安装docker：
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker的安装包：

确保该安装包在当前目录，然后在当前目录创建容器如下：
sudo docker load -i docker_tpuc_dev_v2.2.tar.gz
docker run --privileged --name cvitek2 -v $PWD:/workspace -it sophgo/tpu_dev:latest

运行docker：
docker exec -it cvitek2 bash
### 5.2.2 编译ONNX模型
本节需要如下包：

#### 5.2.2.1 加载tpu-mlir
以下操作需要在docker中进行，需要先加载docker。
解压缩包，并定义环境变量：
tar zxf tpu-mlir_v1.0.1-ga942a1ec-20230402.tar.gz
source tpu_mlir/envsetup.sh
#### 5.2.2.2 准备工作目录
建立pptsm目录，注意需要与tpu_mlir为同级目录；并把模型文件和图片文件都放入pptsm目录中：
mkdir pptsm_model && cd pptsm_model
#cp pptsm.onnx,images .
mkdir workspace && cd workspace
 #### 5.2.2.3 ONNX转MLIR
tpu_mlir中有自定义的python文件model_transform.py将ONNX模型转为MLIR模型，model_transform.py支持的参数如下：

但是注意，model_transform.py只适用于图片输入或npz输入，由于PP-TSM使用的是视频输入，所以此处需要修改model_transform.py，定义专属PP-TSM的预处理函数，或者不修改model_transform.py,但是需要自定义函数，将输入的视频文件转化为npz文件，将npz文件作为PP-TSM使用model_transform.py转换模型时的输入，上述两种方法二选一即可。
1.在model_transform.py中添加PP-TSM的预处理函数：
找到文件tpu_mlir/python/tools/model_transform.py，首先加入PP-TSM的预处理自定义类ppTSM_Inference_helper:
class ppTSM_Inference_helper:
    def __init__(self,
                 num_seg=8,
                 seg_len=1,
                 short_size=256,
                 target_size=224,
                 top_k=1):
                     self.num_seg = num_seg
                     self.seg_len = seg_len
                     self.short_size = short_size
                     self.target_size = target_size
                     self.top_k = top_k

    def scale(self, img, short_size=256):
        h, w, _ = img.shape
        scale_ratio = float(short_size) / min(h, w)
        return cv2.resize(img, (0, 0), fx=scale_ratio, fy=scale_ratio)

    def center_crop(self, img, target_size=224):
        h, w, _ = img.shape
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        return img[start_h:start_h + target_size, start_w:start_w + target_size]

    def random_crop(self, img, target_size=224):
        h, w, _ = img.shape
        start_h = randint(0, h - target_size)
        start_w = randint(0, w - target_size)
        return img[start_h:start_h + target_size, start_w:start_w + target_size]

    def random_flip(self, img, p=0.5):
        if np.random.rand() < p:
            return cv2.flip(img, 1)
        return img

    def normalize(self, img, mean, std):
        img = img.astype(np.float32)
        img /= 255.0
        img -= mean
        img /= std
        return img

    def preprocess(self, input_file):
        assert os.path.isfile(input_file), "{0} not exists".format(input_file)
        results = {'filename': input_file}
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load video
        cap = cv2.VideoCapture(input_file)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        # Sample frames
        num_frames = len(frames)
        seg_size = num_frames // self.num_seg
        sampled_frames = []

        for i in range(self.num_seg):
            idx = seg_size * i + seg_size // 2
            sampled_frames.append(frames[idx])

        # Preprocess frames
        preprocessed_frames = []

        for frame in sampled_frames:
            frame = self.scale(frame, self.short_size)
            frame = self.center_crop(frame, self.target_size)
            frame = self.normalize(frame, img_mean, img_std)
            preprocessed_frames.append(frame)

        # Convert frames to NumPy array
        preprocessed_video = np.stack(preprocessed_frames, axis=0)
        preprocessed_video = np.transpose(preprocessed_video, (0, 3, 1, 2))
        res = np.expand_dims(preprocessed_video, axis=0).copy()
        return [res]

按照如下代码修改类ModelTransformer：
class ModelTransformer(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.converter = BaseConverter()
        self.do_mlir_infer = True

    def cleanup(self):
        file_clean()

    def model_transform(self, mlir_file: str, post_handle_type=""):
        self.mlir_file = mlir_file
        mlir_origin = mlir_file.replace('.mlir', '_origin.mlir', 1)
        file_mark(mlir_origin)
        self.converter.generate_mlir(mlir_origin)
        mlir_opt_for_top(mlir_origin, self.mlir_file, post_handle_type)
        print("Mlir file generated:{}".format(mlir_file))

        self.module_parsered = MlirParser(self.mlir_file)
        self.input_num = self.module_parsered.get_input_num()

    def model_validate(self, file_list: str, tolerance, excepts, test_result):
        in_f32_npz = self.model_name + '_in_f32.npz'
        inputs = dict()
        if len(file_list) == 1 and file_list[0].endswith('.npz'):
            npz_in = np.load(file_list[0])
            for name in self.converter.input_names:
                assert (name in npz_in.files)
                inputs[name] = npz_in[name]
        elif file_list[0].endswith(('.jpg', '.jpeg', '.png')):  #todo add isPicture in util
            ppa = preprocess()
            for i in range(self.input_num):
                pic_path = file_list[i] if i < len(file_list) else file_list[-1]
                file = os.path.expanduser(pic_path)
                ppa.load_config(self.module_parsered.get_input_op_by_idx(i))
                inputs[ppa.input_name] = ppa.run(file)
        elif file_list[0].endswith(('avi')):
            ppa = preprocess()
            helper = ppTSM_Inference_helper()
            for i in range(self.input_num):
                pic_path = file_list[i] if i < len(file_list) else file_list[-1]
                file = os.path.expanduser(pic_path)
                ppa.load_config(self.module_parsered.get_input_op_by_idx(i))
                inputs[ppa.input_name] = helper.preprocess(file)[0]
        else:
            assert (len(file_list) == len(self.converter.input_names))
            for name, file in zip(self.converter.input_names, file_list):
                assert (file.endswith('.npy'))
                inputs[name] = np.load(file)
        np.savez(in_f32_npz, **inputs)

        # original model inference to get blobs of all layer
        ref_outputs = self.origin_inference(inputs)
        if self.do_mlir_infer:
            ref_npz = self.model_name + '_ref_outputs.npz'
            np.savez(ref_npz, **ref_outputs)

            # inference of mlir model
            from tools.model_runner import mlir_inference, show_fake_cmd
            show_fake_cmd(in_f32_npz, self.mlir_file, test_result)
            f32_outputs = mlir_inference(inputs, self.mlir_file)
            np.savez(test_result, **f32_outputs)
            # compare all blobs layer by layers
            f32_blobs_compare(test_result, ref_npz, tolerance, excepts=excepts)
            file_mark(ref_npz)
            else:
            np.savez(test_result, **ref_outputs)

            @abc.abstractmethod
            def origin_inference(self, inputs: dict) -> dict:
            pass

修改后的python文件命名为model_transform_pptsm.py,运行命令进行模型转换：
model_transform_pptsm.py \
--model_name pptsm \
--model_def ../pptsm.onnx \
--resize_dims 256,256 \
--keep_aspect_ratio false \
--mean 2.1178,2.0362,1.8060 \
--scale 4.3668,4.4643,4.4444 \
--pixel_format rgb \
--test_input ./fatigue.avi \
--test_result pptsm_top_outputs.npz \
--mlir pptsm_fp32.mlir
2.将输入的视频文件转化为npz文件：
定义avi_npz.py如下：
import cv2
from random import randint
import os
import numpy as np
class ppTSM_Inference_helper:
    def __init__(self,
                 num_seg=8,
                 seg_len=1,
                 short_size=256,
                 target_size=224,
                 top_k=1):
                     self.num_seg = num_seg
                     self.seg_len = seg_len
                     self.short_size = short_size
                     self.target_size = target_size
                     self.top_k = top_k

    def scale(self, img, short_size=256):
        h, w, _ = img.shape
        scale_ratio = float(short_size) / min(h, w)
        return cv2.resize(img, (0, 0), fx=scale_ratio, fy=scale_ratio)

    def center_crop(self, img, target_size=224):
        h, w, _ = img.shape
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        return img[start_h:start_h + target_size, start_w:start_w + target_size]

    def random_crop(self, img, target_size=224):
        h, w, _ = img.shape
        start_h = randint(0, h - target_size)
        start_w = randint(0, w - target_size)
        return img[start_h:start_h + target_size, start_w:start_w + target_size]

    def random_flip(self, img, p=0.5):
        if np.random.rand() < p:
            return cv2.flip(img, 1)
        return img

    def normalize(self, img, mean, std):
        img = img.astype(np.float32)
        img /= 255.0
        img -= mean
        img /= std
        return img

    def preprocess(self, input_file):
        assert os.path.isfile(input_file), "{0} not exists".format(input_file)
        results = {'filename': input_file}
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load video
        cap = cv2.VideoCapture(input_file)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        # Sample frames
        num_frames = len(frames)
        seg_size = num_frames // self.num_seg
        sampled_frames = []

        for i in range(self.num_seg):
            idx = seg_size * i + seg_size // 2
            sampled_frames.append(frames[idx])

        # Preprocess frames
        preprocessed_frames = []

        for frame in sampled_frames:
            frame = self.scale(frame, self.short_size)
            frame = self.center_crop(frame, self.target_size)
            frame = self.normalize(frame, img_mean, img_std)
            preprocessed_frames.append(frame)

        # Convert frames to NumPy array
        preprocessed_video = np.stack(preprocessed_frames, axis=0)
        preprocessed_video = np.transpose(preprocessed_video, (0, 3, 1, 2))
        res = np.expand_dims(preprocessed_video, axis=0).copy()
        return [res]

def create_npz(video_path, npz_path):
    helper= ppTSM_Inference_helper()
    inputs= dict()
    for file in os.listdir(video_path):
        name = file
        path = os.path.join(video_path,file)
        print(name)
        inputs['data_batch_0'] = helper.preprocess(path)[0]
        # print(inputs)
        x = name.split('.')
        if len(x)>2:
        name=x[0]
        for i in range(1,len(x)-2):
        name=name+x[i]+'_'
        name=name+x[-2]+'.npz'
        name = os.path.join(npz_path, name)
        np.savez(name, **inputs)

        if __name__ == '__main__':
        video_path = '/home/hub/workspace/gq/cvitek/cvitek_tpu_2023/pptsm_model/videos'
        npz_path = '/home/hub/workspace/gq/cvitek/cvitek_tpu_2023/pptsm_model/npzs'
        create_npz(video_path, npz_path)
修改video_path和npz_path即可。将avi转为npz文件之后，执行命令进行模型转换：
model_transform.py \
--model_name pptsm \
--model_def ../pptsm.onnx \
--resize_dims 256,256 \
--keep_aspect_ratio false \
--mean 2.1178,2.0362,1.8060 \
--scale 4.3668,4.4643,4.4444 \
--pixel_format rgb \
--test_input ./fatigue.npz \
--test_result pptsm_top_outputs.npz \
--mlir pptsm_fp32.mlir

转成mlir文件之后，会生成一个${model_name}_in_f32.npz文件，该文件是模型的输入文件。
5.2.2.4 MLIR转BF16模型
tpu_mlir中有自定义的python文件model_deploy.py将MLIR模型转为F32模型，model_deploy.py支持的参数如下：

同样，其也只适用于图片输入或者npz输入，所以此处也需要按照model_transform.py进行同样的处理。由于先将视频输入转化为npz输入比较通用，所以之后的示例都只使用npz作为输入样例。
model_deploy.py \
--mlir pptsm_fp32.mlir \
--quantize BF16 \
--chip cv181x \
--test_input ./pptsm_in_f32.npz \
--test_reference pptsm_top_outputs.npz \
--tolerance 0.8,0.8,0.86 \
--model pptsm_bf16.cvimodel
注意：
1. 关于quantize的设置，其与模型的选择chip有关，我选用的是cv181x作为chip，所以选择BF16，若报错可以修改quantize的值试试。
2. tolerance的值不要设置得太高，否则就会出现如下错误：
[CMD]: model_runner.py --input pptsm_in_f32.npz --model pptsm_bf16.cvimodel --output pptsm_cv181x_bf16_model_outputs.npz
setenv:cv181x
Start TPU Simulator for cv181x
device[0] opened, 4294967296
version: 1.4.0
pptsm Build at 2023-04-21 21:38:02 For platform cv181x
Cmodel: bm_load_cmdbuf
python3: /work/code/tpu_compiler/externals/cviruntime/src/cmodel/cmodel_cmdbuf.cpp:168: virtual bmerr_t CModelCmdbuf::rt_load_cmdbuf(bmctx_t, uint8_t*, size_t, long long unsigned int, long long unsigned int, bool, bm_memory**): Assertion `tdma_sz + tiu_sz <= (int)sz && tdma_sz <= (int)g_tdma_cmdbuf_reserved_size' failed.
Aborted (core dumped)

或：

[Success]: tpuc-opt pptsm_cv181x_bf16_tpu.mlir --mlir-disable-threading --do-extra-opt --strip-io-quant="quant_input=False quant_output=False" --weight-reorder --subnet-divide="dynamic=False" --layer-group="opt=2" --address-assign -o pptsm_cv181x_bf16_final.mlir 
Traceback (most recent call last):
File "/workspace/tpu-mlir/python/tools/model_deploy.py", line 300, in <module>
tool.build_model()
File "/workspace/tpu-mlir/python/tools/model_deploy.py", line 220, in build_model
self.merge_weight,
File "/workspace/tpu-mlir/python/utils/mlir_shell.py", line 130, in mlir_to_model
codegen_param,
UnboundLocalError: local variable 'codegen_param' referenced before assignment
#### 5.2.2.5 MLIR转INT8模型
##### 5.2.2.5.1生成校准表
转INT8模型前需要跑calibration,得到校准表；输入数据的数量根据情况准备100~1000个样例左右。
然后用校准表，生成对称或非对称的cvimodel。如果对称符合需要，一般不建议使用非对称，因为非对称的性能会略差于对称模型。
此处用现有的304个视频转换为的npz文件进行校准，执行calibration:
run_calibration.py pptsm_fp32.mlir \
--dataset ../npzs \
--input_num=304 \
-o pptsm_calibration_table 
5.2.2.5.2编译为INT对称量化模型
model_deploy.py \
--mlir pptsm_fp32.mlir \
--quantize INT8 \
--calibration_table pptsm_calibration_table \
--chip cv181x \
--test_input ./pptsm_in_f32.npz \
--test_reference pptsm_top_outputs.npz \
--tolerance 0.8,0.45 \
--model pptsm_int8_sym.cvimodel
##### 5.2.2.5.3编译为INT非对称量化模型
model_deploy.py \
--mlir pptsm_fp32.mlir \
--quantize INT8 \
--calibration_table pptsm_calibration_table \
--chip cv181x \
--test_input ./pptsm_in_f32.npz \
--test_reference pptsm_top_outputs.npz \
--tolerance 0.8,0.45 \
--model pptsm_int8_asym.cvimodel
#### 5.2.2.6效果对比
##### 5.2.2.6.1自定义PP-TSM的推理代码：
import torch
import argparse
import os
import cv2
import numpy as np
from random import randint
from tools.model_runner import mlir_inference, model_inference, onnx_inference, torch_inference
from utils.preprocess import supported_customization_format

class ppTSM_Inference_helper:
    def __init__(self,
                 num_seg=8,
                 seg_len=1,
                 short_size=256,
                 target_size=224,
                 top_k=1):
                     self.num_seg = num_seg
                     self.seg_len = seg_len
                     self.short_size = short_size
                     self.target_size = target_size
                     self.top_k = top_k

    def scale(self, img, short_size=256):
        h, w, _ = img.shape
        scale_ratio = float(short_size) / min(h, w)
        return cv2.resize(img, (0, 0), fx=scale_ratio, fy=scale_ratio)

    def center_crop(self, img, target_size=224):
        h, w, _ = img.shape
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        return img[start_h:start_h + target_size, start_w:start_w + target_size]

    def random_crop(self, img, target_size=224):
        h, w, _ = img.shape
        start_h = randint(0, h - target_size)
        start_w = randint(0, w - target_size)
        return img[start_h:start_h + target_size, start_w:start_w + target_size]

    def random_flip(self, img, p=0.5):
        if np.random.rand() < p:
            return cv2.flip(img, 1)
        return img

    def normalize(self, img, mean, std):
        img = img.astype(np.float32)
        img /= 255.0
        img -= mean
        img /= std
        return img

    def preprocess(self, input_file):
        assert os.path.isfile(input_file), "{0} not exists".format(input_file)
        results = {'filename': input_file}
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load video
        cap = cv2.VideoCapture(input_file)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        # Sample frames
        num_frames = len(frames)
        seg_size = num_frames // self.num_seg
        sampled_frames = []

        for i in range(self.num_seg):
            idx = seg_size * i + seg_size // 2
            sampled_frames.append(frames[idx])

        # Preprocess frames
        preprocessed_frames = []

        for frame in sampled_frames:
            frame = self.scale(frame, self.short_size)
            frame = self.center_crop(frame, self.target_size)
            frame = self.normalize(frame, img_mean, img_std)
            preprocessed_frames.append(frame)

        # Convert frames to NumPy array
        preprocessed_video = np.stack(preprocessed_frames, axis=0)
        preprocessed_video = np.transpose(preprocessed_video, (0, 3, 1, 2))
        res = np.expand_dims(preprocessed_video, axis=0).copy()
        return [res]


        FATIGUE_CLASSES = ("fatigue", "normal")

        def refine_cvi_output(output):
        new_output = {}
        for k in output.keys():
        if k.endswith("_f32"):
        out = output[k]
        n, c, h, w = out.shape[0], out.shape[1], out.shape[2], out.shape[3]
        new_output[k] = out.reshape(n, c, h, w // 85, 85)
        return new_output

        def main(args):
        video_path = args.dataset
        model_path = args.model

        for video_file in os.listdir(video_path):
        video_file_path = os.path.join(video_path, video_file)
        helper = ppTSM_Inference_helper()
        data = helper.preprocess(video_file_path)[0]
        data = {"data_batch_0": data}
        # output = dict()
        if args.model.endswith('.onnx'):
        output = onnx_inference(data, args.model, False)
elif args.model.endswith('.pt') or args.model.endswith('.pth'):
        output = torch_inference(data, args.model, False)
elif args.model.endswith('.mlir'):
        output = mlir_inference(data, args.model, False)
elif args.model.endswith(".bmodel") or args.model.endswith(".cvimodel"):
        output = model_inference(data, args.model)
        if args.model.endswith(".cvimodel"):
        output = refine_cvi_output(output)
else:
        raise RuntimeError("not support modle file:{}".format(args.model))
        print(f"Video {video_file} is classified as class {output}")

        if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, required=True, help='Path to the folder containing video files')
        parser.add_argument('--model', type=str, required=True, help='Path to the pre-trained ppTSM model')
        args = parser.parse_args()
        main(args)

##### 5.2.2.6.2使用onnx模型进行推理：
python /workspace/tpu-mlir/python/samples/classify_pptsm.py \
--dataset ../videos2 \
--model ../ppTSMv2.onnx
效果：

##### 5.2.2.6.3使用bf16 cvimode进行推理：
python /workspace/tpu-mlir/python/samples/classify_pptsm.py \
--dataset ../videos2 \
--model ./pptsm_bf16.cvimodel
效果：

##### 5.2.2.6.4使用int8对称 cvimode进行推理：
python /workspace/tpu-mlir/python/samples/classify_pptsm.py \
--dataset ../videos2 \
--model ./pptsm_int8_sym.cvimodel
##### 5.2.2.6.5使用int8 非对称 cvimode进行推理
python /workspace/tpu-mlir/python/samples/classify_pptsm.py \
--dataset ../videos2 \
--model ./pptsm_int8_asym.cvimodel
##### 5.2.2.6.6效果对比
类别	视频	模型	分类	置信度
Fatigue	f1.avi	onnx	Fatigue	0.96
	bf16 cvimode	Fatigue	0.95
	int8 aym cvimodel	Fatigue	0.91
	int8 asym cvimodel	Fatigue	0.91
f2.avi	onnx	Fatigue	0.90
	bf16 cvimode	Fatigue	0.90
	int8 aym cvimodel	Fatigue	0.89
	int8 asym cvimodel	Fatigue	0.89
f3.avi	onnx	Fatigue	0.89
	bf16 cvimode	Fatigue	0.89
	int8 aym cvimodel	Fatigue	0.86
	int8 asym cvimodel	Fatigue	0.86
f4.avi	onnx	Fatigue	0.95
	bf16 cvimode	Fatigue	0.95
	int8 aym cvimodel	Fatigue	0.95
	int8 asym cvimodel	Fatigue	0.95
f5.avi	onnx	Fatigue	0.90
	bf16 cvimode	Fatigue	0.90
	int8 aym cvimodel	Fatigue	0.88
	int8 asym cvimodel	Fatigue	0.88
Normal	n1.avi	onnx	Normal	0.98
	bf16 cvimode	Normal	0.98
	int8 aym cvimodel	Normal	0.98
	int8 asym cvimodel	Normal	0.98
n2.avi	onnx	Normal	0.93
	bf16 cvimode	Normal	0.92
	int8 aym cvimodel	Normal	0.95
	int8 asym cvimodel	Normal	0.95
n3.avi	onnx	Normal	0.96
	bf16 cvimode	Normal	0.96
	int8 aym cvimodel	Normal	0.98
	int8 asym cvimodel	Normal	0.98
n4.avi	onnx	Normal	0.99
	bf16 cvimode	Normal	0.99
	int8 aym cvimodel	Normal	0.99
	int8 asym cvimodel	Normal	0.99
n5.avi	onnx	Normal	0.97
	bf16 cvimode	Normal	0.96
	int8 aym cvimodel	Normal	0.98
	int8 asym cvimodel	Normal	0.98
## 5.3 EVB板编译
### 5.3.1 环境配置
#### 5.3.1.1 软件安装
需要用到Ubuntu（推荐使用Ubuntu 20.04 LTS）作为系统环境，可选择装载Ubuntu系统的主机或在VM Ware中创建虚拟机。
##### 5.3.1.1.1 在VM Ware安装Ubuntu
##### 5.3.1.1.2 PuTTy安装
##### 5.3.1.1.3 MobaXterm安装
#### 5.3.1.2连接开发板
1.Windows系统
1.串口通信连接：

连接之后，打开设备管理器，如果电脑已有FT232R USB UART的驱动，

没有则会显示其他设备： 

之后参照教程https://blog.csdn.net/u013767242/article/details/79571463下载驱动即可。
驱动下载好时，双击USB Serial Port(COM3),在端口设置处设置每秒位数为115200（默认值是9600，这个数值是无法正常连接EVB板的），数据位8，停止位1，流控制无。

现在就可以使用串口通信工具进行连接了，我用了PuTTY和MobaXterm，都能连接成功。其中PuTTY设置如下：



最后点击Open就可以进行连接了。

MobaXterm连接：



2.网线+ssh连接
注意：如果在虚拟机上连接，需要将虚拟机的网络设置为桥接模式，并且分别手动配置虚拟机和开发板的网络：
ifconfig eth0 192.168.137.25 netmask 255.255.255.0 up
sudo ifconfig ens33 192.168.137.128 netmask 255.255.255.0 up

服务器端：
关闭防火墙：
sudo ufw disable
sudo iptables -F

教程：https://blog.csdn.net/A601023332/article/details/111876651
https://blog.csdn.net/ping_devil/article/details/106592993
或者直接将开发板和PC连入同一个路由器中，详情参考下面的Linux系统的网线+ssh连接。
注意：如果网线+ssh无法登录到开发板，或者在上述教程中，给开发板分配ip时使用arp -a：

很明显没有看到给开发板分配的动态随机IP，这个时候我们需要先进行串口通信连接，再运行arp -a就可以看到分配的动态ip了：

然后再使用网线+ssh连接，关闭串口通信也可以了。

2. Linux系统（此处使用的Ｕbuntu20.04 LST） 
1.串口通信连接
参考教程：
串口通信：https://blog.csdn.net/Electrical_IT/article/details/108125482
相关错误及解决方案：https://blog.csdn.net/xfwyzxw/article/details/121747258
https://blog.csdn.net/qq_38125389/article/details/103619500
https://blog.csdn.net/wzz953200463/article/details/115704223
实操：
ls /dev/tty*
#如果此处输出中有/dev/ttyUSBx,那么就表明有UART驱动，否则需要参照上述教程下载驱动

sudo apt-get install minicom
sudo minicom -s

选择Serial port setup,按enter键进入，修改A，E，F项的值如下图所示：

选择Save setup as dfl，按enter键:

再选择Exit，按enter键,之后再按任意键，则会进入如下界面，即串口通信连接成功了。

此时键入可能会发现有回显，先Ctrl+A，再键入Z，出现菜单：

可以看到本地回显local echo开关对应的是E，那么先Ctrl+A，再键入E，即可将回显关闭。
2. 网线+ssh连接
直接将开发板和Ubuntu主机接入到同一个路由器中，路由器会自动给开发板分配一个ip，先进行串口通信使用ifconfig命令查看开发板的ip地址：

可以看到eth0网卡的ip是192.168.1.10,使用ssh root@192.168.1.10连接即可。
#### 5.3.1.3 获取资源包
参考：https://github.com/sophgo/cvi_mmf_sdk
##### 5.3.1.3.1 获取cvi_mmf_sdk
使用git clone命令：
git clone https://github.com/sophgo/cvi_mmf_sdk.git
##### 5.3.1.3.2 获取编译工具链
使用wget命令：
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/03/07/16/host-tools.tar.gz
##### 5.3.1.3.3解压工具链并链接到SDK目录
tar xvf host-tools.tar.gz
cd cvi_mmf_sdk/
ln -s ../host-tools ./
##### 5.3.1.3.4安装编译依赖工具包
sudo apt install pkg-config
sudo apt install build-essential
sudo apt install ninja-build
sudo apt install automake
sudo apt install autoconf
sudo apt install libtool
sudo apt install wget
sudo apt install curl
sudo apt install git
sudo apt install gcc
sudo apt install libssl-dev
sudo apt install bc
sudo apt install slib
sudo apt install squashfs-tools
sudo apt install android-sdk-libsparse-utils
sudo apt install android-sdk-ext4-utils
sudo apt install jq
sudo apt install cmake
sudo apt install python3-distutils
sudo apt install tclsh
sudo apt install scons
sudo apt install parallel
sudo apt install ssh-client
sudo apt install tree
sudo apt install python3-dev
sudo apt install python3-pip
sudo apt install device-tree-compiler
sudo apt install libssl-dev
sudo apt install ssh
sudo apt install cpio
sudo apt install squashfs-tools
sudo apt install fakeroot
sudo apt install libncurses5
sudo apt install flex
sudo apt install bison
注意：cmake版本最低要求3.16.5，但是用一般安装命令：
sudo apt install cmake
安装的并不是cmake最新版本。需要参考教程：https://www.cnblogs.com/yibeimingyue/p/15604692.html。
#### 5.3.1.4 编译
cd cvi_mmf_sdk/
source build/cvisetup.sh
#需要根据开发板的型号进行修改
defconfig cv1811c_wevb_0006a_spinor
build_all
编译成功后将在install目录看到生成的image：

#### 5.3.1.5烧录
● 接好EVB板的串口线
● 将SD卡格式化成FAT32格式
● 将install目录下的image放入SD卡根目录
注意：一定要保证下列文件位于SD卡根目录。
.
├── boot.spinor
├── data.spinor
├── fip.bin
├── fw_payload_uboot.bin
├── partition_spinor.xml
└── rootfs.spinor
● 将SD卡插入的SD卡槽中
● 将平台重新上电，开机自动进入烧录，烧录过程log如下：
Hit any key to stop autoboot:  0
##Resetting to default environment
Start SD downloading...
mmc1 : finished tuning, code:60
465408 bytes read in 11 ms (40.3 MiB/s)
mmc0 : finished tuning, code:27
switch to partitions #1, OK
mmc0(part 1) is current device

MMC write: dev # 0, block # 0, count 2048 ... 2048 blocks written: OK in 17 ms (58.8 MiB/s)

MMC write: dev # 0, block # 2048, count 2048 ... 2048 blocks written: OK in 14 ms (71.4 MiB/s)
Program fip.bin done
mmc0 : finished tuning, code:74
switch to partitions #0, OK
mmc0(part 0) is current device
64 bytes read in 3 ms (20.5 KiB/s)
Header Version:1
2755700 bytes read in 40 ms (65.7 MiB/s)

MMC write: dev # 0, block # 0, count 5383 ... 5383 blocks written: OK in 64 ms (41.1 MiB/s)
64 bytes read in 4 ms (15.6 KiB/s)
Header Version:1
13224 bytes read in 4 ms (3.2 MiB/s)

MMC write: dev # 0, block # 5760, count 26 ... 26 blocks written: OK in 2 ms (6.3 MiB/s)
64 bytes read in 4 ms (15.6 KiB/s)
Header Version:1
11059264 bytes read in 137 ms (77 MiB/s)

MMC write: dev # 0, block # 17664, count 21600 ... 21600 blocks written: OK in 253 ms (41.7 MiB/s)
64 bytes read in 3 ms (20.5 KiB/s)
Header Version:1
4919360 bytes read in 65 ms (72.2 MiB/s)

MMC write: dev # 0, block # 158976, count 9608 ... 9608 blocks written: OK in 110 ms (42.6 MiB/s)
64 bytes read in 4 ms (15.6 KiB/s)
Header Version:1
10203200 bytes read in 128 ms (76 MiB/s)

MMC write: dev # 0, block # 240896, count 19928 ... 19928 blocks written: OK in 228 ms (42.7 MiB/s)
Saving Environment to MMC... Writing to MMC(0)... OK
mars_c906#
● 烧录成功，拔掉SD卡，重新给板子上电，进入系统。
注意：断电后再登录时一定要拔掉SD卡，或者保证SD卡不再有需要烧录的文件：
.
├── boot.spinor
├── data.spinor
├── fip.bin
├── fw_payload_uboot.bin
├── partition_spinor.xml
└── rootfs.spinor
## 5.4 部署算法到EVB板
本节需要cvitek_tpu_sdk包，并且需要在docker环境中运行。
### 5.4.1 激活编译环境
解压包并声明环境变量：
export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..
### 5.4.2 交叉编译
自定义PP-TSM的推理代码如下：
#include <stdio.h>
#include <fstream>
#include <string>
#include <cviruntime.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdlib>
#include<iostream>
const int NUM_SEGMENT = 8; // 根据需求进行修改
const int HEIGHT = 224;
const int WIDTH = 224;
const int CHANNELS = 3;

std::vector<cv::Mat> read_images(const std::string& folder_path) {
    std::vector<cv::Mat> images;
    for (int i = 0; i < NUM_SEGMENT; ++i) {
        std::string file_path = folder_path + "/frame_" + std::to_string(i+1) + ".jpg";
        cv::Mat img = cv::imread(file_path, cv::IMREAD_COLOR);
        //    cv::Mat img = cv::imread(file_path);

        //    std::cout<<img<<std::endl;
        if (img.empty()) {
            std::cerr << "Cannot read image: " << file_path << std::endl;
            continue;
        }
        images.push_back(img);
    }
    return images;
}

void ppTSM_preprocessing(const std::vector<cv::Mat>& input_images, CVI_TENSOR* input_tensor) {
    int channels = CHANNELS;
    cv::Mat resized_image;
    int crop_size = 224;
    input_tensor->shape.dim[0] = NUM_SEGMENT;
    input_tensor->shape.dim[1] = CHANNELS;
    input_tensor->shape.dim[2] = HEIGHT;
    input_tensor->shape.dim[3] = WIDTH;
    input_tensor->shape.dim_size = 4;
    input_tensor->fmt = CVI_FMT_FP32;
    input_tensor->mem_size = NUM_SEGMENT * channels * HEIGHT * WIDTH * sizeof(float);
    input_tensor->sys_mem = new uint8_t[input_tensor->mem_size];

    float* input_data = reinterpret_cast<float*>(input_tensor->sys_mem);

    for (int i = 0; i < NUM_SEGMENT; ++i) {
        //    int frame_idx = std::min(i * segment_duration + segment_duration / 2, frame_count - 1);
        //    cv::Mat frame = input_images[frame_idx];
        int frame_idx = i;
        cv::Mat frame = input_images[frame_idx];
        cv::resize(frame, resized_image, cv::Size(crop_size, crop_size));
        resized_image.convertTo(resized_image, CV_32F, 1 / 255.0);

        // 数据减均值并除以标准差
        cv::Mat mean = cv::Mat(resized_image.size(), CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
        cv::Mat std_dev = cv::Mat(resized_image.size(), CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
        resized_image = (resized_image - mean) / std_dev;


        // 将每个预处理后的图像连接到输入张量
        int idx = i * channels * crop_size * crop_size;
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < crop_size; ++h) {
                for (int w = 0; w < crop_size; ++w) {
                    input_data[idx++] = resized_image.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }
}

int main(int argc, char** argv) {
if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <image_folder> <model_path>" << std::endl;
    return -1;
}

std::string folder_path = argv[1];
std::string model_path = argv[2];

// 读取图片
std::vector<cv::Mat> images = read_images(folder_path);

// 设置模型
CVI_MODEL_HANDLE model;
CVI_NN_RegisterModel(model_path.c_str(), &model);

// 获取输入输出张量
CVI_TENSOR *inputs, *outputs;
int32_t input_num, output_num;
CVI_NN_GetInputOutputTensors(model, &inputs, &input_num, &outputs, &output_num);

// 预处理
ppTSM_preprocessing(images, &inputs[0]);

// 推理
CVI_NN_Forward(model, inputs, input_num, outputs, output_num);

// 
float* output_data = static_cast<float*>(CVI_NN_TensorPtr(&outputs[0]));
int output_size = CVI_NN_TensorCount(&outputs[0]);

// 获取最大概率的类别
int max_index = std::distance(output_data, std::max_element(output_data, output_data + output_size));
float max_prob = output_data[max_index];

std::cout << "Predicted class: " << max_index << ", probability: " << max_prob << std::endl;

// 清理
printf("------\n");
CVI_NN_CleanupModel(model);
printf("CVI_NN_CleanupModel succeeded\n");

return 0;
}

上述代码中包含的数据结构为cviruntime的自定义数据结构和自定义函数。
进入之前创建的文件夹pptsm_model，进行如下操作：
cd pptsm_model_exec
mkdir build_soc
cd build_soc

cmake \
DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_C_FLAGS_RELEASE=-O3 -DCMAKE_CXX_FLAGS_RELEASE=-O3 \
-DCMAKE_TOOLCHAIN_FILE=$TPU_SDK_PATH/cmake/toolchain-riscv64-linux-musl-x86_64.cmake \
-DTPU_SDK_PATH=$TPU_SDK_PATH \
-DOPENCV_PATH=$TPU_SDK_PATH/opencv \
-DCMAKE_INSTALL_PREFIX=../install_samples \
..

cmake --build . --target install
# 查看编译后的运行文件
cd ..
cd install_samples/bin/
ls

如果报错如下，说明$PATH中没有包含riscv64-unknown-linux-musl-g++的目录：
CMake Error at CMakeLists.txt:3 (project):
  The CMAKE_CXX_COMPILER:

    riscv64-unknown-linux-musl-g++

  is not a full path and was not found in the PATH.

在根目录使用find命令查找该文件，再设置PATH目录：
root@131f100abfee:/workspace# find . -name riscv64-unknown-linux-musl-gcc
./host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-gcc


root@131f100abfee:/workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin# export PATH=$PATH:$PWD
### 5.4.3 运行
#### 5.4.3.1 登录开发板
使用ssh连接或者串口连接都可。
#### 5.4.3.2 挂载目录
##### 5.4.3.2.1 服务器（PC）端：
安装nfs-kernel-server:
sudo apt-get install nfs-kernel-server
选择/建立mount文件夹，注意一定要把刚才生成的可执行文件包含进来。
修改/etc/exports文件，添加如下内容：
/home/nfs_server \ *(rw,sync,no_subtree_check,no_root_squash)
restart nfs服务：
sudo /etc/init.d/rpcbind restart
sudo /etc/init.d/nfs-kernel-server restart
##### 5.4.3.2.2 EVB板端：
mount -t nfs 192.168.137.128:/home/orla/workspace/cvitek  /mnt/nfs -o nolock

mount -t nfs 192.168.137.128:/home/hub/workspace/gq/cvitek/cvitek_tpu_2023/  /mnt/nfs -o nolock
注意：PC端和EVB板必须连接在同一局域网内。
#### 5.4.3.3运行
# 声明环境变量
export TPU_SDK_PATH=$PWD/cvitek_tpu_sdk
cd cvitek_tpu_sdk && source ./envs_tpu_sdk.sh && cd ..
# 进入脚本路径
cd pptsm_model_exec/install_samples/bin
## 执行脚本
./cvi_pptsm_bf16 ../../fatigue ../../pptsm_bf16.cvimodel
运行截图：

注意：
