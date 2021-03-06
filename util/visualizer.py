import numpy as np
import os
import sys
import ntpath
import time
from subprocess import Popen, PIPE
from . import util, html, utilBitDepth

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(opt, webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        opt                      -- stores all the experiment flags; need to be subclass of BaseOptions
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    # Grabbing the correct tensor2im function according to the dataset.
    if opt.dataset_mode == 'np_array_aligned':
        tensor2im = utilBitDepth.tensor2im
    else: 
        tensor2im = util.tensor2im

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if opt.saveNpyImages:
            utilBitDepth.saveVisualsAsNpy(im_data, os.path.join(os.path.dirname(save_path), *['..', 'npyImages', os.path.basename(save_path)]).replace('.png',''))
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if opt.dataset_mode == 'np_array_aligned':
            self.tensor2im = utilBitDepth.tensor2im
            self.save_image =util.save_image
        else: 
            self.tensor2im = util.tensor2im
            self.save_image = util.save_image

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        if opt.validate:
        # create a logging file to store validation losses
            self.val_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'val_loss_log.csv')
            if os.path.exists(self.val_log_name) and opt.continue_train:
                pass
            else:
                with open(self.val_log_name, 'w+') as val_log_file:
                    val_log_file.write('time,epoch,iters,val_loss,val_std\n')

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                # h, w = next(iter(visuals.values())).shape[:2]
                h, w = len(visuals) // ncols + bool(len(visuals) % ncols), ncols 
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = self.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                black_image = np.zeros_like(image_numpy.transpose([2, 0, 1]))
                while idx % ncols != 0:
                    images.append(black_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = self.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = self.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                self.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = self.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_current_losses_ewma(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        """We assume this is called after plot_current_losses and thus the 
        plot data should already exist"""
        # if not hasattr(self, 'ewma_win_id'):
        #     self.ewma_win_id = str(self.get_next_win_id())
        
        self.plot_data['Y_ewma'] = np.apply_along_axis(util.ewma_halflife, 0, np.array(self.plot_data['Y']), self.opt.display_ewma_halflife)

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y_ewma']),
                opts={
                    'title': self.name + f' enwa loss over time (halflife = {self.opt.display_ewma_halflife})',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id + 3)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_validation_losses(self, epoch, counter_ratio, val_losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (Dict) -- validation loss dict with fields mean_val_loss and std_val_loss
        """
        
        loss_crit = self.opt.validation_loss

        val_loss_dict = {
            loss_crit : val_losses['mean_val_loss'],
            loss_crit + ' + sd' : val_losses['mean_val_loss'] + val_losses['std_val_loss'],
            loss_crit + ' - sd' : val_losses['mean_val_loss'] - val_losses['std_val_loss'],
        }

        if not hasattr(self, 'val_plot_data'):
            self.val_plot_data = {
             'X': [],
             'loss': [],
             'loss_ewma': [],
             'loss+sd': [],
             'loss-sd': [],
             'legend': [loss_crit, loss_crit + ' ewma', loss_crit + ' + sd', loss_crit + ' - sd']
             }

        self.val_plot_data['X'].append(epoch + counter_ratio)
        self.val_plot_data['loss'].append(val_loss_dict[loss_crit])
        self.val_plot_data['loss+sd'].append(val_loss_dict[loss_crit + ' + sd'])
        self.val_plot_data['loss-sd'].append(val_loss_dict[loss_crit + ' - sd'])
        self.val_plot_data['loss_ewma'] = np.apply_along_axis(util.ewma_halflife, 0, np.array(self.val_plot_data['loss']), self.opt.display_ewma_halflife)

        try:
            self.vis.line(
                X=np.stack([np.array(self.val_plot_data['X'])] * len(self.val_plot_data['legend']), 1),
                Y=np.stack(np.array([
                    self.val_plot_data['loss'],
                    self.val_plot_data['loss_ewma'],
                    self.val_plot_data['loss+sd'],
                    self.val_plot_data['loss-sd'],
                ]), 1),
                opts={
                    'title': self.name + f'Validation loss ({loss_crit})',
                    'legend': self.val_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id + 5)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_validation_losses(self, epoch, iters, losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (Dict) -- validation loss dict with fields mean_val_loss and std_val_loss
        """
        now = time.strftime("%c")
        message = ','.join([str(elem) for elem in [now, epoch, iters, float(losses['mean_val_loss']), float(losses['std_val_loss'])]])

        print(f'Validation loss {losses["mean_val_loss"]} +/- {losses["std_val_loss"]}')

        with open(self.val_log_name, "a") as val_log_file:
            val_log_file.write(message + '\n')  # save the message

    def get_next_win_id(self):
        win_id = 1
        while self.vis.win_exists(str(win_id)):
            win_id += 1
        return win_id