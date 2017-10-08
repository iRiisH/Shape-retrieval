import numpy as np
import vispy as vp
import parser
import sys
import vispy.plot as plt
import vispy.scene as scn
import cv2
import os
import math
import progressbar


def record_to_str(rc):
    s = str(rc[0])
    for i in range(1, len(rc)):
        s += ' ' + str(rc[i])
    return s


def best_view():
    vp.app.use_app('pyqt5')

    canvas = scn.SceneCanvas(keys='interactive', size=(600, 600), show=True, bgcolor='white')

    view = canvas.central_widget.add_view()
    view.camera = scn.TurntableCamera()

    dtype = [
        ('model_id', int),
        ('azimuth', float),
        ('elevation', float),
        ('class', int)]

    n = 1815
    cur_model_id = 0
    record_filename = 'best_worst_direction.txt'

    records = parser.load(record_filename)
    vs, fs, fc = parser.parse(cur_model_id)
    vs -= np.mean(vs, axis=0)
    mesh = scn.visuals.Mesh(vertices=vs, faces=fs, vertex_colors=fc, parent=view.scene)

    @canvas.connect
    def on_key_press(event):
        global view, mesh, cur_model_id, n, record_filename, records
        print('-----')
        ch = event.text
        if ch == 'n' or ch == 'p':
            cur_model_id = (cur_model_id + (1 if ch == 'n' else n-1)) % n
            vs, fs, fc = parser.parse(cur_model_id)
            mesh.set_data(vertices=vs, faces=fs, vertex_colors=fc)
            print(('next' if ch == 'n' else 'previous')+' model -> model id: ' + str(cur_model_id))
        if ch == 'b' or ch == 'w':
            az = view.camera.azimuth
            el = view.camera.elevation
            cl = int(ch == 'b')
            records += [(cur_model_id, az, el, cl)]
            print('record '+('best' if ch == 'b' else 'worst') + ' direction')
            print('model id, azumith, elevation: '+str(cur_model_id)+' '+str(az)+' '+str(el))
        if ch == 'u':
            if len(records) > 0:
                print('undo: pop last record '+record_to_str(records[-1]))
                print('existing records: '+str(len(records)))
                del records[-1]
        if ch == 's':
            f = open(record_filename, 'w')
            s = ''
            for rc in records:
                s += record_to_str(rc) + '\n'
            f.write(s)
            f.close()
            print('save records successfully')

    if sys.flags.interactive != 1:
        vp.app.run()


def euclidian2spherical(x, y, z):
    """
    converts v = (x, y, z) st |v| = 1 into (elevation, azimuth)
    """
    elevation = math.asin(z)
    azimuth = 0.
    if abs(z) != 1.:
        azimuth = math.acos(x/math.cos(elevation))
        if y < 0:
            azimuth += math.pi
    return elevation, azimuth


def save_views(n_views):
    """
    loads the meshes using the points of view generated by Lloyd's relaxation,
    and saves the associated image
    """
    vp.app.use_app('pyqt5')
    n = 1815
    bar = progressbar.ProgressBar()
    for cur_model_id in bar(range(n)):
        root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        model_dir = os.path.join(root_dir, 'data/imgs/m{}/'.format(cur_model_id))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        axis = []
        # load view vector coordinates
        with open(os.path.join(root_dir, 'data/views/{}.txt'.format(n_views)), 'r') as f:
            for line in f:
                x, y, z = list(map(float, line[:-1].split(' ')))
                elevation, azimuth = euclidian2spherical(x, y, z)
                axis.append((elevation, azimuth))
        # load images associated to these axis
        for nax, ax in enumerate(axis):
            elevation, azimuth = ax
            dst_file = os.path.join(model_dir, 'm{}_{}.png'.format(cur_model_id, nax))
            canvas = scn.SceneCanvas(keys='interactive', size=(600, 600), bgcolor='white')
            view = canvas.central_widget.add_view()
            view.camera = scn.TurntableCamera()
            # dtype = [
            #     ('model_id', int),
            #     ('azimuth', float),
            #     ('elevation', float),
            #     ('class', int)]
            # records = parser.load(record_filename)
            vs, fs, fc = parser.parse(cur_model_id)
            vs -= np.mean(vs, axis=0)
            scn.visuals.Mesh(vertices=vs, faces=fs, vertex_colors=fc, parent=view.scene)
            view.camera.scale_factor = 1.5
            view.camera.elevation = (180./math.pi)*elevation
            view.camera.azimuth = (180./math.pi)*azimuth
            # print(view.camera.scale_factor)
            # print(view.camera.azimuth)
            # print(view.camera.elevation)
            img = canvas.render()
            cv2.imwrite(dst_file, img)


if __name__ == '__main__':
    save_views(7)
