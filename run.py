from Normalize import Normalize, Blur, Normalize_TF
import timm
import scipy.stats as st
from attack_method import *
from torch.nn import DataParallel
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='/workspace/input_dir/dev.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='/workspace/input_dir/images/', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='/workspace/output_dir/', help='Input directory with images.')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')

parser.add_argument("--max_epsilon", type=float, default=20.0, help="Maximum size of adversarial perturbation.")

parser.add_argument("--num_iter_set", type=int, default=40, help="Number of iterations.")

parser.add_argument("--image_width", type=int, default=500, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=500, help="Height of each input images.")
# parser.add_argument("--image_resize", type=int, default=[560, 620, 680, 740, 800], help="Height of each input images.")
parser.add_argument("--image_resize", type=int, default=560, help="Height of each input images.")

parser.add_argument("--batch_size", type=int, default=2, help="How many images process at one time.")

parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--amplification", type=float, default=1.5, help="To amplifythe step size.")
parser.add_argument("--prob", type=float, default=0.7, help="probability of using diverse inputs.")

opt = parser.parse_args()

torch.backends.cudnn.benchmark = True
# loss_fn_vgg = lpips.LPIPS(net='vgg')
transforms = T.Compose([T.ToTensor()])


def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2

def project_noise(x, stack_kern, kern_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding = (kern_size, kern_size), groups=3)
    return x

stack_kern, kern_size = project_kern(3)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def graph(x, gt, x_min, x_max, **models):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    alpha_beta = alpha * opt.amplification
    gamma = alpha_beta

    eff = models["eff"]
    dense = models['dense']
    res = models['res50']
    res101 = models['res101']
    # wide = models['wide']
    dense169 = models['dense169']
    vgg = models['vgg']
    # lpipsLoss = models["lpips"]

    res101.zero_grad()
    eff.zero_grad()
    dense.zero_grad()
    dense169.zero_grad()
    res.zero_grad()
    # wide.zero_grad(True)
    vgg.zero_grad()
    # lpipsLoss.zero_grad(True)


    # x.requires_grad = True
    adv = x.clone()
    adv = adv.cuda()
    adv.requires_grad = True
    amplification = 0.0
    pre_grad = torch.zeros(adv.shape).cuda()

    for i in range(num_iter):
        if i == 0:
            adv = F.conv2d(adv, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
            adv = clip_by_tensor(adv, x_min, x_max)
            adv = V(adv, requires_grad = True)
        output1 = 0
        output1 += dense(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
        output1 += res(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
        output1 += res101(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
        output1 += dense169(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
        output1 += vgg(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
        output1 += eff(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
        # output1 += wide(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./7
        loss1 = F.cross_entropy(output1 * 1.5, gt, reduction="none")

        output3 = 0
        output3 += dense(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
        output3 += res(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
        output3 += res101(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
        output3 += dense169(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
        output3 += vgg(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
        output3 += eff(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
        # output3 += wide(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./7
        loss3 = F.cross_entropy(output3 * 1.5, gt, reduction="none")

        output4 = 0
        output4 += dense(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
        output4 += res(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
        output4 += res101(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
        output4 += dense169(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
        output4 += vgg(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
        output4 += eff(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
        # output4 += wide(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./7
        loss4 = F.cross_entropy(output4 * 1.5, gt, reduction="none")

        output5 = 0
        output5 += dense(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
        output5 += res(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
        output5 += res101(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
        output5 += dense169(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
        output5 += vgg(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
        output5 += eff(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
        # output5 += wide(F.interpolate(ensemble_input_diversity(adv + pre_grad, 4), (256, 256), mode='bilinear')) * 1./7
        loss5 = F.cross_entropy(output5 * 1.5, gt, reduction="none")

        loss = (loss1 + loss3 + loss4 + loss5) / 4.0
        # loss = loss1
        # print('loss = ', loss)
        # lpipsL = lpipsLoss(x, adv, normalize=True)[:, 0, 0, 0]
        # coeff = (lpipsL * 10).detach()
        # lpipsL = lpipsL.clamp(0.2, 10.)
        # print('lpipsl = ', coeff * lpipsL)
        # loss = loss - coeff * lpipsL

        loss.mean().backward()
        noise = adv.grad.data
        pre_grad = adv.grad.data
        noise = F.conv2d(noise, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)

        # MI-FGSM
        # noise = noise / torch.abs(noise).mean([1,2,3], keepdim=True)
        # noise = momentum * grad + noise
        # grad = noise

        # PI-FGSM
        amplification += alpha_beta * torch_staircase_sign(noise, 1.5625)
        cut_noise = clip_by_tensor(abs(amplification) - eps, 0.0, 10000.0) * torch.sign(amplification)
        projection = alpha * torch_staircase_sign(project_noise(cut_noise, stack_kern, kern_size), 1.5625)

        # staircase sign method (under review) can effectively boost the transferability of adversarial examples, and we will release our paper soon.
        pert = (alpha_beta * torch_staircase_sign(noise, 1.5625) + 0.5 * projection) * 0.75
        # adv = adv + pert * (1-mask) * 1.2 + pert * mask * 0.8
        adv = adv + pert
        # print(mask.max())
        # print(mask.min())
        # exit()
        # adv = adv + alpha * torch_staircase_sign(noise, 1.5625)
        adv = clip_by_tensor(adv, x_min, x_max)
        adv = V(adv, requires_grad = True)

    return adv.detach()


def main():

    dense = DataParallel(torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                models.densenet121(pretrained=True).eval())).cuda()
    res = DataParallel(torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                models.resnet50(pretrained=True).eval())).cuda()
    res101 = DataParallel(torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                models.resnet101(pretrained=True).eval())).cuda()
    # wide = DataParallel(torch.nn.Sequential(Normalize(opt.mean, opt.std),
    #                             models.wide_resnet101_2(pretrained=True).eval())).cuda()
    vgg = DataParallel(torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                models.vgg19(pretrained=True).eval())).cuda()
    dense169 = DataParallel(torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                models.densenet169(pretrained=True).eval())).cuda()
    eff = DataParallel(nn.Sequential(Normalize_TF(), timm.create_model('tf_efficientnet_b5', pretrained=True).eval())).cuda()

    # lpipsLoss = DataParallel(lpips.LPIPS(net='vgg', verbose=False).cuda())

    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    # sum_dense, sum_res, sum_res101, sum_dense169, sum_vgg, sum_xception, sum_adv, sum_eff, sum_wide = 0,0,0,0,0,0,0,0, 0

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    iter = 0
    for images, name, gt_cpu in tqdm(data_loader):
        iter += 1
        gt = gt_cpu.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
        adv_img = graph(images, gt, images_min, images_max, eff=eff, dense=dense, res50=res, res101=res101, dense169=dense169, vgg=vgg)

        for i in range(len(adv_img)):
            save_img(opt.output_dir + '{}'.format(name[i]), adv_img[i].detach().permute(1,2,0).cpu())

    #     with torch.no_grad():
    #         sum_dense += (dense(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         # sum_res += (res(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         # sum_wide_res += (wide_res(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_res += (res(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_res101 += (res101(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_vgg += (vgg(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_dense169 += (dense169(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_eff += (eff(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_wide += (wide(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #
    #         if iter % 20 == 0:
    #             batch_size = len(adv_img)
    #             print('dense = {:.2%}'.format(sum_dense / (batch_size * iter)))
    #             print('res = {:.2%}'.format(sum_res / (batch_size * iter)))
    #             print('res101 = {:.2%}'.format(sum_res101 / (batch_size * iter)))
    #             print('wide = {:.2%}'.format(sum_wide / (batch_size * iter)))
    #             print('vgg = {:.2%}'.format(sum_vgg / (batch_size * iter)))
    #             print('dense169 = {:.2%}'.format(sum_dense169 / (batch_size * iter)))
    #             print('sum_eff = {:.2%}'.format(sum_eff / (batch_size * iter)))
    #
    # # print('dense = {:.2%}'.format(sum_dense / 5000.0))
    # # print('res = {:.2%}'.format(sum_res / 5000.0))
    # # print('wide_res = {:.2%}'.format(sum_wide_res / 5000.0))
    # # print('next = {:.2%}'.format(sum_next / 5000.0))
    # # print('vgg = {:.2%}'.format(sum_vgg / 5000.0))
    # # print('xception = {:.2%}'.format(sum_xception / 5000.0))
    # # print('sum_adv = {:.2%}'.format(sum_adv / 5000.0))
    # print('res = {:.2%}'.format(sum_res / 5000))
    # print('res101 = {:.2%}'.format(sum_res101 / 5000))
    # print('wide = {:.2%}'.format(sum_wide / 5000))
    # print('dense169 = {:.2%}'.format(sum_dense169 / 5000))
    # print('vgg = {:.2%}'.format(sum_vgg / 5000))
    # print('sum_eff = {:.2%}'.format(sum_eff / 5000))

    # score_fid = cal_fid(opt.input_dir, opt.output_dir)
    # score_lpips = cal_lpips(opt.input_dir, opt.output_dir)
    # final_score = 100 * score_fid * score_lpips
    # print('score_lpips: ', score_lpips)
    # print('score_fid:', score_fid)
    # print("final score: score_ASR * ", final_score)


if __name__ == '__main__':
    main()
