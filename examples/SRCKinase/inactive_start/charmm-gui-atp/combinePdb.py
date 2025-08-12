#!/usr/bin/env python
import sys,os
import argparse
import textwrap

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-option', dest="option")
    parser.add_argument('-chains', nargs='*', dest="chains")
    parser.add_argument('-orig', dest="orig")
    parser.add_argument('-resn', dest="resn")
    parser.add_argument('-multi', nargs='*', dest="multi")

    inputarg = parser.parse_args()
    option     = inputarg.option
    chains   = inputarg.chains
    orig     = inputarg.orig
    resn     = inputarg.resn
    multi   = inputarg.multi

    # READ ligand information and Mapping

    with open("ligandrm.yml",'r') as stream:
        rm_info = yaml.load(stream, Loader=yaml.FullLoader)

    rm_info["newresn"] = resn.upper()

    orgpdb = rm_info['orgfile'].split(".")[0]+".pdb"

    mapping = {}
    mappingcnt = []

    uploadDic = {}
    origDic = {}


    cnt = 0
    for line in open("upload.crd",'r'):
        if '0.0000000000' in line:
            line = line.strip().split()
            atomn = line[3]
            coorx = float(line[4])
            coory = float(line[5])
            coorz = float(line[6])
            uploadDic[cnt] = { "atomn" : atomn,
                                 "coorx" : "%.2f" % coorx,
                                 "coory" : "%.2f" % coory,
                                 "coorz" : "%.2f" % coorz }
            mappingcnt.append(atomn)
            cnt += 1

    cnt = 0
    for line in open(orgpdb,'r'):
        if line.startswith("ATOM"):
            atomn = line[11:16].strip()
            coorx = float(line[30:38])
            coory = float(line[38:46])
            coorz = float(line[46:54])
            origDic[cnt] = { "atomn" : atomn,
                             "coorx" : "%.2f" % coorx,
                             "coory" : "%.2f" % coory,
                             "coorz" : "%.2f" % coorz }
            cnt += 1
    for i in range(0,len(uploadDic)):
        for j in range(0,len(origDic)):
            if (uploadDic[i]["coorx"] == origDic[j]["coorx"]) and (uploadDic[i]["coory"] == origDic[j]["coory"]) and (uploadDic[i]["coorz"] == origDic[j]["coorz"]):
                mapping[uploadDic[i]["atomn"]] = origDic[j]["atomn"]

    # READ HEADER

    with open("header.yml",'r') as stream:
        header = yaml.load(stream, Loader=yaml.FullLoader)

    # HEADER
    string  = "%-10s%-60s\n" % ("HEADER", "CHARMM-GUI Ligand Reader & Modeler")

    # TITLE
    if ( header['title'] ):
        htitle = header['title'].strip().translate(None,"|\n")
        ltitle = textwrap.wrap(htitle, 60)
        for i in range(0,len(ltitle)):
            string  += "%-10s%-60s\n" % ("TITLE", ltitle[i])
    #EXPDTA
    if (header['expdata']):
        string += "%-10s%-60s\n" % ("EXPDTA",header['expdata'].strip().translate(None,"|\n"))

    #SSBOND
    for i in range(0,len(header['ssbonds'])):
        segid1 = header['ssbonds'][i][0]['segid'][3]
        resid1 = int(header['ssbonds'][i][0]['resid'])
        segid2 = header['ssbonds'][i][1]['segid'][3]
        resid2 = int(header['ssbonds'][i][1]['resid'])
        string += "%-6s %3i CYS %1s %4s    CYS %1s %4s\n" % ("SSBOND",i+1, segid1, resid1, segid2, resid2)

    #CRYST1
    if len(header['crystal']['geometry']) > 0:
        if ( header['crystal']['geometry']['a'] and
             header['crystal']['geometry']['b'] and
             header['crystal']['geometry']['c'] and
             header['crystal']['geometry']['alpha'] and
             header['crystal']['geometry']['beta'] and
             header['crystal']['geometry']['gamma'] and
             header['crystal']['geometry']['p'] and
             header['crystal']['geometry']['p1'] and
             header['crystal']['geometry']['p1'] and
             header['crystal']['geometry']['p2'] and
             header['crystal']['geometry']['p3']):

            string += "%6s%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %s %s %s %s%5i\n" % (
                        "CRYST1",
                        header['crystal']['geometry']['a'],
                        header['crystal']['geometry']['b'],
                        header['crystal']['geometry']['c'],
                        header['crystal']['geometry']['alpha'],
                        header['crystal']['geometry']['beta'],
                        header['crystal']['geometry']['gamma'],
                        header['crystal']['geometry']['p'],
                        header['crystal']['geometry']['p1'],
                        header['crystal']['geometry']['p1'],
                        header['crystal']['geometry']['p2'],
                        header['crystal']['geometry']['p3']
                        )

    if (option == "LoadPDBID" or option == "UploadPDB"):
        orig_file   = orig.split(".")[0]
        orig_parser = orig.split(".")[0].split("_")
        modi_chain = orig_parser[-1]
        pdbid       = orig_file.split(modi_chain)[0]
        pdbid       = pdbid[:-1]

        for i in range(0,len(chains)):
            fname = "%s_%s.pdb" % (pdbid,chains[i])

            if (chains[i] != modi_chain):   # if the chain is not modified chain
                if (multi != ""):
                    if chains[i] in multi:      # if the chain is the same residue as modified residue
                        switch = True
                        for line in open(fname, 'r'):
                            if (switch and line.startswith("ATOM")):
                                tempchain = line[21]
                                tempresid = line[22:26]
                                tempsegid = line[72:76]
                                switch = False

                        for line in open(fname.split(".")[0]+"_fin.pdb", 'r'):
                            if line.startswith("ATOM"):
                                string += line[0:21] + "%1s" % tempchain + "%4s" % tempresid + line[26:72] + "%4s" % tempsegid + line[77:] + "\n"
                        string += "TER\n"
                    else:
                        for line in open(fname,'r'):
                            if line.startswith("ATOM"):
                                string += line
                        string += "TER\n"
            else:
                switch = True
                for line in open(fname,'r'):
                    if (switch and line.startswith("ATOM")):
                        tempchain = line[21]
                        tempresid = line[22:26]
                        tempsegid = line[72:76]
                        switch = False

                fin = open("ligandrm.pdb",'r')

                for line in fin:
                    if line.startswith("ATOM"):
                        line = line[0:21] + "%1s" % tempchain + "%4s" % tempresid + line[26:72] + "%4s" % tempsegid + line[77:] + "\n"
                        string += line
                string += "TER\n"
        string += "END\n"
        fout = open("%s_modified.pdb" % pdbid,'w')
        fout.write(string)

    elif (option == "combinatorial"):
        orig_file   = orig.split(".")[0]
        orig_parser = orig.split(".")[0].split("_")
        modi_chain = orig_parser[-1]
        pdbid       = orig_file.split(modi_chain)[0]
        pdbid       = pdbid[:-1]

        dircnt = int(1)

        while(os.path.exists("ld%s" % dircnt)):
            cpstr = string

            for i in range(0,len(chains)):
                fname = "%s_%s.pdb" % (pdbid,chains[i])

                if (chains[i] != modi_chain): # if the chain is not modified chain
                    if chains[i] in multi:      # if the chain is the same residue as modified residue
                        switch = True
                        for line in open(fname, 'r'):
                            if (switch and line.startswith("ATOM")):
                                tempchain = line[21]
                                tempresid = line[22:26]
                                tempsegid = line[72:76]
                                switch = False

                        for line in open("ld%s/" % dircnt+fname.split(".")[0]+"_fin.pdb", 'r'):
                            if line.startswith("ATOM"):
                                cpstr += line[0:21] + "%1s" % tempchain + "%4s" % tempresid + line[26:72] + "%4s" % tempsegid + line[77:] + "\n"
                        cpstr += "TER\n"
                    else:
                        for line in open(fname,'r'):
                            if line.startswith("ATOM"):
                                cpstr += line
                        cpstr += "TER\n"
                else:
                    switch = True
                    for line in open(fname,'r'):
                        if (switch and line.startswith("ATOM")):
                            tempchain = line[21]
                            tempresid = line[22:26]
                            tempsegid = line[72:76]
                            switch = False

                    fin = open("ld%s/lig.pdb" % dircnt,'r')

                    for line in fin:
                        if line.startswith("ATOM"):
                            line = line[0:21] + "%1s" % tempchain + "%4s" % tempresid + line[26:72] + "%4s" % tempsegid + line[77:] + "\n"
                            cpstr += line
                    cpstr += "TER\n"
            cpstr += "END\n"
            fout = open("ld%s/%s_modified.pdb" % (dircnt, pdbid),'w')
            fout.write(cpstr)

            dircnt += 1


