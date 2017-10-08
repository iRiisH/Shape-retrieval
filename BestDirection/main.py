import numpy as np
import vispy as vp
import parser
import sys
import vispy.plot as plt
import vispy.scene as scn
vp.app.use_app('pyqt5')

canvas = scn.SceneCanvas(keys='interactive', size=(600, 600), show=True, bgcolor='white')

view = canvas.central_widget.add_view()
view.camera = scn.TurntableCamera()

dtype = [
    ('model_id', int),
    ('azimuth', float),
    ('elevation', float),
    ('class', int)]


def record_to_str(rc):
    s = str(rc[0])
    for i in range(1, len(rc)):
        s += ' '+str(rc[i])
    return s

n = 1815
cur_model_id = 0
record_filename = 'best_worst_direction.txt'

records = parser.load(record_filename)
vs, fs, fc = parser.parse(cur_model_id)
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

if __name__ == '__main__' and sys.flags.interactive != 1:
    vp.app.run()
