
import numpy as np
import time
import argparse


def read_centroids(fname_cent):
    """Read the centroid data file"""
    with open(fname_cent, 'r') as f0:
        for cent in f0:
            if cent.startswith('1'):
                centroids = np.array([int(cent.split(' ')[0]), float(cent.split(' ')[1]), float(cent.split(' ')[2]), 0.0])
                while True:
                    try:
                        cent = next(f0)
                        centroids = np.vstack([centroids, [int(cent.split(' ')[0]), float(cent.split(' ')[1]), float(cent.split(' ')[2]), 0.0]])
                    except StopIteration:
                        break
    return centroids


def read_input(fname_inp, f2, centroids):
    """Read the input file created by Neper"""
    elsets = {}

    with open(fname_inp, 'r') as f1:
        for line in f1:
            if line.startswith('*Part'):
                f2.write(line)
            elif line.startswith('*Node'):
                cline = next(f1)
                nodes = np.array([int(cline.split(',')[0]), float(cline.split(',')[1]), float(cline.split(',')[2]), float(cline.split(',')[3])])
                while True:
                    cline = next(f1)
                    try:
                        isinstance(int(cline.split(',')[0]), int)
                        nodes = np.vstack([nodes, [int(cline.split(',')[0]), float(cline.split(',')[1]), float(cline.split(',')[2]), float(cline.split(',')[3])]])
                    except ValueError:
                        break
            elif line.startswith('*Element, type=CPE3'):
                cline = next(f1)
                elements_CPE3 = np.array([int(cline.split(',')[0]), int(cline.split(',')[1]), int(cline.split(',')[2]), int(cline.split(',')[3])])
                while True:
                    cline = next(f1)
                    try:
                        isinstance(int(cline.split(',')[0]), int)
                        elements_CPE3 = np.vstack([elements_CPE3, [int(cline.split(',')[0]), int(cline.split(',')[1]), int(cline.split(',')[2]), int(cline.split(',')[3])]])
                    except ValueError:
                        break
            elif line.startswith('*Element, type=COH2D4'):
                cline = next(f1)
                elements_COH2D4 = np.array([int(cline.split(',')[0]), int(cline.split(',')[1]), int(cline.split(',')[2]), int(cline.split(',')[3]),
                                            int(cline.split(',')[4])])
                while True:
                    cline = next(f1)
                    try:
                        isinstance(int(cline.split(',')[0]), int)
                        elements_COH2D4 = np.vstack([elements_COH2D4, [int(cline.split(',')[0]), int(cline.split(',')[1]), int(cline.split(',')[2]), int(cline.split(',')[3]),
                                                                       int(cline.split(',')[4])]])
                    except ValueError:
                        break
            elif line.startswith('*Elset'):
                for j in range(1, len(centroids[:,0])+1, 1):
                    if line.rstrip() == ('*Elset, elset=face'+'%i' %j):
                        cline = next(f1)
                        elsets['Elset_face'+'%i' %j] = cline.strip().split(',')
                        while True:
                            cline = next(f1)
                            try:
                                isinstance(int(cline.split(',')[0]), int)
                                elsets['Elset_face'+'%i' %j].extend(cline.strip().split(','))
                            except ValueError:
                                break
                    else:
                        continue
    return (nodes, elements_CPE3, elements_COH2D4, elsets)


def translate_nodes(nodes, elsets, centroids, elements_CPE3, sfactor):
    """Translate the nodes within the individual grains"""
    pointlist = []
    multipointlist = []
    nsets = {}
    tpsets = {}

    # Find multiple nodes to indentify intersections:
    for nline in nodes[:, 1:4]:
        # Yields positions in nodes array, not node indices!
        pointlist.append(np.where((nodes[:, 1:4] == nline).all(axis=1)))
    multipointlist = [x[0] for x in pointlist if len(x[0]) > 3]

    # Remove redundant entries:
    for t in multipointlist:
        for u in np.where([np.array_equal(t, x) for x in multipointlist])[0][1:]:
            multipointlist[u] = []
    multipointredlist = [x for x in multipointlist if len(x) > 0]

    # Store node indices:
    for h in range(0, len(multipointredlist), 1):
        tpsets['Multipoint'+'%i' %(h+1)] = nodes[[x for x in multipointredlist[h]]]

    # Translate nodes:
    for k in range(1, len(centroids[:, 0])+1, 1):
        cur_elementinds = [int(x) for x in elsets['Elset_face'+'%i' %k] if x]
        cur_elements = np.zeros((1, 4))

        for it1 in cur_elementinds:
            cur_elements = np.vstack([cur_elements, [elements_CPE3[np.where(elements_CPE3[:, 0] == it1)[0][0]]]])
        cur_elements = np.delete(cur_elements, 0, axis=0)
        cur_nodeinds = cur_elements[:, 1:5]
        cur_nodeinds = list(set(list(cur_nodeinds.flatten())))
        cur_nodes = np.zeros((1, 4))

        for it2 in cur_nodeinds:
            cur_nodes = np.vstack([cur_nodes, [nodes[np.where(nodes[:, 0] == int(it2))[0][0]]]])
        cur_nodes = np.delete(cur_nodes, 0, axis=0)
        centre = np.ones((len(cur_nodes[:, 0]), 3))
        centre = centroids[k-1, 1:4]*centre
        dxdydz = cur_nodes[:, 1:4] - centre
        dist = np.sqrt(np.sum(dxdydz**2, axis = 1))
        exeyez = np.divide(dxdydz, np.tile(dist.reshape(len(dist),1), 3))
        exeyez[np.isnan(exeyez)] = dxdydz[np.isnan(exeyez)]
        cur_nodes_write = cur_nodes.copy()
        cur_nodes_write[:, 1:4] = cur_nodes[:, 1:4] - sfactor*dist.reshape(len(dist), 1)*exeyez
        nsets['Nset'+'%i' %k] = cur_nodes_write
    return (multipointlist, multipointredlist, tpsets, nsets)


def update_coordinates(nsets, nodes):
    """Update coordinates of translated nodes"""
    new_entries = []
    new_entries.append([z for z in nsets.values()])
    new_entries = np.concatenate(new_entries[0], axis=0)
    box_length_x = max(nodes[:,1])
    box_length_y = max(nodes[:,2])
    nodes_write = nodes.copy()

    for it3 in new_entries[:, 0]:
        if [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][1] == 0.0 and [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][2] == 0.0:
            continue
        elif [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][1] == 0.0 and [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][2] == box_length_y:
            continue
        elif [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][1] == box_length_x and [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][2] == 0.0:
            continue
        elif [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][1] == box_length_x and [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][2] == box_length_y:
            continue
        elif [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][1] == 0.0:
            [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][2:4] = [new_entries[np.where(new_entries[:, 0] == int(it3))[0][0]]][0][2:4]
        elif [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][2] == 0.0:
            [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][1] = [new_entries[np.where(new_entries[:, 0] == int(it3))[0][0]]][0][1]
        elif [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][1] == box_length_x:
            [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][2:4] = [new_entries[np.where(new_entries[:, 0] == int(it3))[0][0]]][0][2:4]
        elif [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][2] == box_length_y:
            [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]][0][1] = [new_entries[np.where(new_entries[:, 0] == int(it3))[0][0]]][0][1]
        else:
            [nodes_write[np.where(nodes_write[:, 0] == int(it3))[0][0]]] = [new_entries[np.where(new_entries[:, 0] == int(it3))[0][0]]]
    return nodes_write


def assure_period(nodes_write):
    """Assure mesh periodicity at domain boundary"""
    nodes_x0 = nodes_write[nodes_write[:, 1] == 0]
    nodes_x1 = nodes_write[nodes_write[:, 1] == max(nodes_write[:, 1])]
    nodes_y0 = nodes_write[nodes_write[:, 2] == 0]
    nodes_y1 = nodes_write[nodes_write[:, 2] == max(nodes_write[:, 2])]
    nodes_x0 = nodes_x0[np.argsort(nodes_x0[:, 2], 0)]
    nodes_x1 = nodes_x1[np.argsort(nodes_x1[:, 2], 0)]
    nodes_x1[:, 2] = nodes_x0[:, 2].copy()
    nodes_y0 = nodes_y0[np.argsort(nodes_y0[:, 1], 0)]
    nodes_y1 = nodes_y1[np.argsort(nodes_y1[:, 1], 0)]
    nodes_y1[:, 1] = nodes_y0[:, 1].copy()

    for it4 in nodes_x1[:,0]:
        [nodes_write[np.where(nodes_write[:, 0] == int(it4))[0][0]]][0][2] = nodes_x1[np.where(nodes_x1[:, 0] == int(it4))][0, 2]

    for it5 in nodes_y1[:,0]:
        [nodes_write[np.where(nodes_write[:, 0] == int(it5))[0][0]]][0][1] = nodes_y1[np.where(nodes_y1[:, 0] == int(it5))][0, 1]
    return nodes_write


def generate_elements(multipointredlist, multipointlist, nodes_write, nodes, tpsets):
    """Generate missing elements at grain boundary intersections"""
    for l in range(0, len(multipointredlist), 1):
        centrenodelist = []
        transnodelist = []
        for t in multipointlist[l]:
            if np.array_equal(nodes_write[t], nodes[t]):
                centrenodelist.append(t)
            elif not np.array_equal(nodes_write[t], nodes[t]):
                transnodelist.append(t)

        if len(multipointlist[l]) == 4:

            # Compute incenter:
            A = nodes_write[transnodelist[0], 1:3]
            B = nodes_write[transnodelist[1], 1:3]
            C = nodes_write[transnodelist[2], 1:3]

            # Store coordinates of incenter:
            incenter = (A*np.linalg.norm(B-C) + B*np.linalg.norm(C-A) + C*np.linalg.norm(A-B))/(np.linalg.norm(B-C) + np.linalg.norm(C-A) + np.linalg.norm(A-B))
            nodes_write[centrenodelist[0]] = [nodes_write[centrenodelist[0]][0], incenter[0], incenter[1], 0.0]

        # Store coordinates of center node:
        tpsets['Multipoint_center'+'%i' %(l+1)] = nodes_write[centrenodelist[0]]

        # Store adjacent nodes:
        tpsets['Multipoint_transnodes'+'%i' %(l+1)] = nodes_write[[x for x in transnodelist]]

    # Sort adjacent nodes in counter-clockwise order:
    for t in range(0, len(multipointredlist), 1):
        centrend = np.ones((len(tpsets['Multipoint_transnodes'+'%i' %(t+1)][:, 0]), 2))
        centrend = tpsets['Multipoint_center'+'%i' %(t+1)][1:3]*centrend
        p_nodes = tpsets['Multipoint_transnodes'+'%i' %(t+1)][:, 1:3] - centrend
        p_base = [1.0, 0.0]*centrend
        arglist = []
        for i in range(0, len(p_nodes[:, 0]), 1):
            argi = np.arccos(np.dot(p_nodes[i], p_base[i])/(np.linalg.norm(p_nodes[i])*np.linalg.norm(p_base[i])))
            if p_nodes[i, 1] >= 0.0:
                arglist.append(argi)
            elif p_nodes[i, 1] < 0.0:
                arglist.append(2*np.pi - argi)
            ordlist = np.argsort(arglist)

            # Store ordered adjacent nodes:
            tpsets['Multipoint_transnodes_ord'+'%i' %(t+1)] = tpsets['Multipoint_transnodes'+'%i' %(t+1)][ordlist]

    # Store missing elements:
    elements_CPE3_write = []
    for f in range(0, len(multipointredlist), 1):
        for h in range(0, len(tpsets['Multipoint_transnodes_ord'+'%i' %(f+1)][:, 0]), 1):
            try:
                elements_CPE3_write.append([h, int(tpsets['Multipoint_center'+'%i' %(f+1)][0]), int(tpsets['Multipoint_transnodes_ord'+'%i' %(f+1)][h, 0]), int(tpsets['Multipoint_transnodes_ord'+'%i' %(f+1)][h+1, 0])])
            except IndexError:
                elements_CPE3_write.append([h, int(tpsets['Multipoint_center'+'%i' %(f+1)][0]), int(tpsets['Multipoint_transnodes_ord'+'%i' %(f+1)][h, 0]), int(tpsets['Multipoint_transnodes_ord'+'%i' %(f+1)][0, 0])])
    return (nodes_write, tpsets, elements_CPE3_write)


def write_modinput(f2, nodes_write, elements_COH2D4, elements_CPE3, elements_CPE3_write, fname_inp):
    """Write modified input file"""
    f2.write('**\n')
    f2.write('*Node\n')
    for lineout in nodes_write:
        f2.write('%i, %15.12f, %10.12f, %10.12f\n' %(int(lineout[0]), lineout[1], lineout[2], lineout[3]))

    f2.write('**\n')
    f2.write('*Element, type=COH2D4\n')
    for lineout in elements_COH2D4:
        f2.write('%i, %i, %i, %i, %i\n' %(int(lineout[0]), int(lineout[1]), int(lineout[2]), int(lineout[3]), int(lineout[4])))

    f2.write('**\n')
    f2.write('*Element, type=CPE3\n')
    for lineout in elements_CPE3:
        f2.write('%i, %i, %i, %i\n' %(int(lineout[0]), int(lineout[1]), int(lineout[2]), int(lineout[3])))

    indc = max(elements_CPE3[:, 0])+1000000
    elsetlist = []
    for k in range(0, len(elements_CPE3_write[:]), 1):
        f2.write('%i, %i, %i, %i\n' %(int(indc+k), int(elements_CPE3_write[k][1]), int(elements_CPE3_write[k][2]), int(elements_CPE3_write[k][3])))
        elsetlist.append(int(indc+k))

    elsetlist = [str(x)+',' for x in elsetlist]
    linebreak = range(16, int(len(elsetlist)+np.ceil(len(elsetlist)/15)), 16)
    linebreak = [x-1 for x in linebreak]
    for x in linebreak:
        elsetlist.insert(x, '\n')

    f2.write('**\n')
    f2.write('*Elset, elset=newelements\n')
    f2.write(''.join(str(x) for x in elsetlist))
    f2.write('\n')

    # Copy remainder of input file:
    flag = False
    f2.write('**\n')
    for line in open(fname_inp, 'r'):
        if line.rstrip() == '*Elset, elset=face1':
            flag = True
            f2.write(line)
        elif flag == True:
            f2.write(line)
    f2.close()


def main():
    parser = argparse.ArgumentParser(prog="PyPolyCris2D.py", usage="Run PyPolyCris2D as: python %(prog)s -i INPUTFILENAME -c CENTROIDFILENAME -p MPER -s SFACTOR",
                                     description="PyPolyCris2D generates a 2D polycrystal microstructure mesh from Neper input data with finite-thickness interface elements at the grain boundaries.")
    parser.add_argument("-i", dest="inputfilename",
                        help="import polycrystal microstructure mesh from INPUTFILENAME")
    parser.add_argument("-c", dest="centroidfilename",
                        help="read coordinates of grain centroids from CENTROIDFILENAME")
    parser.add_argument("-p", dest="mper", default="n", choices=["n", "y"],
                        help="indicate whether mesh is periodic (MPER = 'y') or (by default) not periodic (MPER = 'n')")
    parser.add_argument("-s", dest="sfactor", type=float, default="0.1",
                        help="SFACTOR (default is 0.1), which should be selected from the interval (0, 0.5), controls the amount of shrinkage for each grain")
    args = parser.parse_args()

    # Assign input parameters:
    fname_inp = args.inputfilename
    fname_cent = args.centroidfilename
    mper = args.mper
    sfactor = args.sfactor

    # Initilize output file:
    f2 = open(fname_inp[:-4]+'.modified.inp', 'w')

    # Read grain centroids:
    print("\nReading files...\n")
    time.sleep(1)
    centroids = read_centroids(fname_cent)

    # Read Neper input file:
    (nodes, elements_CPE3, elements_COH2D4, elsets) = read_input(fname_inp, f2, centroids)

    # Translate nodes within the grains:
    (multipointlist, multipointredlist, tpsets, nsets) = translate_nodes(nodes, elsets, centroids, elements_CPE3, sfactor)

    # Update node coordinates:
    print("Generating finite-thickness interface elements...\n")
    time.sleep(1)
    nodes_write = update_coordinates(nsets, nodes)

    # Assure mesh periodicity:
    if mper == "y":
        nodes_write = assure_period(nodes_write)
    elif mper == "n":
        pass

    # Generate missing elements:
    (nodes_write, tpsets, elements_CPE3_write) = generate_elements(multipointredlist, multipointlist, nodes_write, nodes, tpsets)

    # Write modified input file:
    write_modinput(f2, nodes_write, elements_COH2D4, elements_CPE3, elements_CPE3_write, fname_inp)

    # Print notification:
    print("Modified input file for SFACTOR = %f successfully generated.\n" %sfactor)
    time.sleep(1)


if __name__ == "__main__":
    main()
