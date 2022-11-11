#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VMEC2SPEC.py


import math
import xarray
import numpy as np
from typing import List, Dict


# permeability of vacuum
mu0 = 4 * math.pi * 1e-7


def VMECout2SPEC(VMEC_output: str, SPEC_input: str, interfaceLabel: List[float], fluxLabel: str = "toroidal", lconstraint: int = 2, lrad: int = 8) -> None:
    """
    This function creates a SPEC input namelist from a VMEC output file. 
    Args:
        VMEC_output: The VMEC outputput file. 
        SPEC_input: The SPEC input file. 
        interfaceLabel: The normalized magnetic flux in subvolumes. 
        fluxLabel : The surface label shoulde be "toroidal" or "poloidal". 
        lconstraint: selects constraints.
        lrad: Chebyshev resolution.
    """

    try:
        VMECout = xarray.open_dataset(VMEC_output)
    except:
        raise FileExistsError(
            "Please cheak your argument. The first argument should be a VMEC output. "
        )
    if (abs(interfaceLabel[0]) > 1e-10) or (abs(interfaceLabel[-1] - 1) > 1e-10):
        raise ValueError(
            "The first interfaceLabel should be 0 and the last interfacelabel should be 1. "
        )
    if not all([interfaceLabel[i] < interfaceLabel[i+1] for i in range(len(interfaceLabel)-1)]):
        raise ValueError(
            "The interfaceLabel should be strictly monotonically increasing. "
        )

    VMEC_ns = int(VMECout["ns"].values)
    VMEC_tflux = VMECout["phi"].values
    # VMEC_pflux = VMECout["chi"].values * -1
    VMEC_pflux = VMECout["chi"].values
    VMEC_tpflux2 = VMEC_tflux * VMEC_pflux
    VMEC_psi = VMEC_tflux / np.pi / 2
    VMEC_chi = VMEC_pflux / np.pi / 2
    if fluxLabel == "toroidal":
        flux_label = VMEC_tflux / VMEC_tflux[-1]
    elif fluxLabel == "poloidal":
        flux_label = VMEC_pflux / VMEC_pflux[-1]
    else:
        raise ValueError(
            "The flux label should be toroidal or poloidal, cheak the flux label. "
        )
    # VMEC_iota = VMECout["iotaf"].values
    VMEC_iota = VMECout["iotaf"].values * -1
    VMEC_g = [abs(VMECout["gmnc"].values[i][0]) for i in range(VMEC_ns)]
    VMEC_jpol = VMECout["jcuru"].values
    VMEC_jtor = VMECout["jcurv"].values
    VMEC_gamma = VMECout["gamma"].values
    VMEC_nfp = int(VMECout["nfp"].values)
    VMEC_mpol = int(VMECout["mpol"].values)
    VMEC_ntor = int(VMECout["ntor"].values)
    VMEC_pressure = VMECout["presf"].values
    VMEC_im = VMECout["xm"].values
    VMEC_in = -1 * VMECout["xn"].values / VMEC_nfp
    # VMEC_in = VMECout["xn"].values / VMEC_nfp
    VMEC_rmnc = VMECout["rmnc"].values
    VMEC_zmns = -1 * VMECout["zmns"].values
    for i in range(len(VMEC_im)):
        if VMEC_im[i] == 0 and VMEC_in[i] == 0:
            VMEC_zmns[i] *= -1

    nvol = len(interfaceLabel) - 1
    # interfacePsi = np.interp(interfaceLabel, flux_label, VMEC_psi)
    datas = {}
    datas["phiedge"] = VMEC_tflux[-1]
    datas["curtor"] = mu0 * 2 * np.pi * integrate(
        flux_label, VMEC_jtor * VMEC_g, flux_label[0], flux_label[-1]
    )
    datas["curpol"] = mu0 * 2 * np.pi * integrate(
        flux_label, VMEC_jpol * VMEC_g, flux_label[0], flux_label[-1]
    )
    datas["gamma"] = VMEC_gamma
    datas["nfp"] = VMEC_nfp
    datas["nvol"] = nvol
    datas["mpol"] = VMEC_mpol
    datas["ntor"] = VMEC_ntor
    datas["lrad"] = [int(lrad) for i in range(nvol)]
    datas["lconstraint"] = int(lconstraint)
    datas["tflux"] = np.interp(interfaceLabel[1:], flux_label,
                               VMEC_tflux) / VMEC_tflux[-1]
    datas["pflux"] = np.interp(interfaceLabel[1:], flux_label,
                               VMEC_pflux) / VMEC_tflux[-1]
    if fluxLabel == "toroidal":
        datas["helicity"] = [(
            4 * math.pi * math.pi * integrate(
                flux_label, VMEC_iota * VMEC_psi - VMEC_chi, interfaceLabel[i], interfaceLabel[i+1]
            ) + np.interp(interfaceLabel[i+1], flux_label, VMEC_tpflux2)
            - np.interp(interfaceLabel[i], flux_label, VMEC_tpflux2)
        ) for i in range(nvol)]
    elif fluxLabel == "poloidal":
        datas["helicity"] = [(
            4 * math.pi * math.pi * integrate(
                flux_label, VMEC_psi - np.divide(VMEC_chi, VMEC_iota), interfaceLabel[i], interfaceLabel[i+1]
            ) + np.interp(interfaceLabel[i+1], flux_label, VMEC_tpflux2)
            - np.interp(interfaceLabel[i], flux_label, VMEC_tpflux2)
        ) for i in range(nvol)]
    datas["pressure"] = [((integrate(
        flux_label, VMEC_pressure * VMEC_g, interfaceLabel[i], interfaceLabel[i + 1])
        / integrate(flux_label, VMEC_g, interfaceLabel[i], interfaceLabel[i + 1])
    )) for i in range(nvol)]
    datas["ivolume"] = [(mu0 * 2 * math.pi * integrate(
        flux_label, VMEC_g * VMEC_jtor, interfaceLabel[i], interfaceLabel[i + 1]))for i in range(nvol)]
    datas["mu"] = [(datas["ivolume"][i] / datas["tflux"][i])
                   for i in range(nvol)]
    datas["isurf"] = [0 for i in range(nvol)]
    datas["iota"] = np.interp(interfaceLabel, flux_label, VMEC_iota)
    datas["rac"] = VMEC_rmnc[0, :]
    datas["zas"] = VMEC_zmns[0, :]
    datas["im"] = VMEC_im
    datas["in"] = VMEC_in
    datas["rbc"] = VMEC_rmnc[-1, :]
    datas["zbs"] = VMEC_zmns[-1, :]
    datas["interface_rc"] = np.zeros([len(interfaceLabel), len(VMEC_im)])
    datas["interface_zs"] = np.zeros([len(interfaceLabel), len(VMEC_im)])
    for j in range(len(VMEC_im)):
        datas["interface_rc"][:, j] = np.interp(
            interfaceLabel, flux_label, VMEC_rmnc[:, j]
        )
        datas["interface_zs"][:, j] = np.interp(
            interfaceLabel, flux_label, VMEC_zmns[:, j]
        )

    writeSPECInput(SPEC_input, datas)

    return


def integrate(baseX: List[float] or np.ndarray, baseY: List[float] or np.ndarray, xLeft: float, xRight: float) -> float:
    if xLeft < min(baseX):
        raise ValueError(
            "The value of xLeft is out of range. "
        )
    if xRight > max(baseX):
        raise ValueError(
            "The value of xRight is out of range. "
        )
    if len(baseX) != len(baseY):
        raise ValueError(
            "The length of baseX and baseY should be equal. "
        )
    nums = len(baseX)
    ans = 0
    base = sorted([i for i in zip(baseX, baseY)], key=lambda k: [k[0], k[1]])
    index = 0
    while index < nums:
        if base[index][0] > xLeft:
            tempXLeft = xLeft
            tempYLeft = np.interp(xLeft, [base[i][0] for i in range(nums)], [
                                  base[i][1] for i in range(nums)])
            while base[index][0] < xRight:
                tempXRight = base[index][0]
                tempYRight = base[index][1]
                ans += (tempXRight - tempXLeft) * (tempYRight + tempYLeft) / 2
                tempXLeft = base[index][0]
                tempYLeft = base[index][1]
                index += 1
            tempXRight = xRight
            tempYRight = np.interp(xRight, [base[i][0] for i in range(nums)], [
                                   base[i][1] for i in range(nums)])
            ans += (tempXRight - tempXLeft) * (tempYRight + tempYLeft) / 2
            break
        else:
            index += 1
    return ans


def writeSPECInput(SPEC_input: str, datas: Dict) -> None:
    """
    Write SPEC input file. 
    """

    nvol = datas["nvol"]
    ninterface = nvol +1

    file = open(SPEC_input, "w")

    # physicslist
    file.write("&physicslist\n")

    file.write("    " + "{:12} = {:d}".format("igeometry",   3) + "\n")
    file.write("    " + "{:12} = {:d}".format("istellsym",   1) + "\n")
    file.write("    " + "{:12} = {:d}".format("lfreebound",  0) + "\n")

    file.write("    " + "{:12} = {:.5e}".format("phiedge", datas["phiedge"]) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("curtor", datas["curtor"]) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("curpol", datas["curpol"]) + "\n")
    
    file.write("    " + "{:12} = {:.5e}".format("gamma", datas["gamma"]) + "\n")
    
    file.write("    " + "{:12} = {:d}".format("nfp",  datas["nfp"]) + "\n")
    file.write("    " + "{:12} = {:d}".format("nvol", datas["nvol"]) + "\n")
    file.write("    " + "{:12} = {:d}".format("mpol", datas["mpol"]) + "\n")
    file.write("    " + "{:12} = {:d}".format("ntor", datas["ntor"]) + "\n")
    
    file.write("    {:12} = ".format("lrad"))
    for i in range(nvol):
        file.write("{:<11d}".format(datas["lrad"][i]))
        file.write("  ")
    file.write("\n")
    
    file.write("    " + "{:12} = {:d}".format("lconstraint", int(datas["lconstraint"])) + "\n")
    
    file.write("    {:12} = ".format("tflux"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["tflux"][i]))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("pflux"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["pflux"][i]))
        file.write("  ")
    file.write("\n")
    
    file.write("    {:12} = ".format("helicity"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["helicity"][i]))
        file.write("  ")
    file.write("\n")
    
    file.write("    " + "{:12} = {:.5e}".format("pscale", 1) + "\n")
    file.write("    {:12} = ".format("pressure"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["pressure"][i]))
        file.write("  ")
    file.write("\n")
    file.write("    " + "{:12} = {:d}".format("ladiabatic", 0) + "\n")
    file.write("    {:12} = ".format("adiabatic"))
    for i in range(nvol):
        file.write("{:.5e}".format(0))
        file.write("  ")
    file.write("\n")

    file.write("    {:12} = ".format("mu"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["mu"][i]))
        file.write("  ")
    file.write("\n")

    file.write("    {:12} = ".format("ivolume"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["ivolume"][i]))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("isurf"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["isurf"][i]))
        file.write("  ")
    file.write("\n")
    
    file.write("    {:12} = ".format("pl"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("ql"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("pr"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("qr"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("iota"))
    for i in range(ninterface):
        file.write("{:.5e}".format(datas["iota"][i]))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("lp"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("lq"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("rp"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("rq"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("oita"))
    for i in range(ninterface):
        file.write("{:.5e}".format(datas["iota"][i]))
        file.write("  ")
    file.write("\n")

    file.write("    {:12} = ".format("rac"))
    for i in range(datas["ntor"] + 1):
        file.write("{:.5e}".format(datas["rac"][i]))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("zas"))
    for i in range(datas["ntor"] + 1):
        file.write("{:.5e}".format(datas["zas"][i]))
        file.write("  ")
    file.write("\n")
    for i in range(len(datas["rbc"])):
        file.write("    {:12} = ".format("rbc(" + str(int(datas["in"][i])) + ", " + str(int(datas["im"][i])) + ")"))
        file.write("{:.5e}".format(datas["rbc"][i]))
        file.write("    {:12} = ".format("zbs(" + str(int(datas["in"][i])) + ", " + str(int(datas["im"][i])) + ")"))
        file.write("{:.5e}".format(datas["zbs"][i]))
        file.write("    {:12} = ".format("rbs(" + str(int(datas["in"][i])) + ", " + str(int(datas["im"][i])) + ")"))
        file.write("{:.5e}".format(0))
        file.write("    {:12} = ".format("zbc(" + str(int(datas["in"][i])) + ", " + str(int(datas["im"][i])) + ")"))
        file.write("{:.5e}".format(0))
        file.write("\n")

    file.write("    " + "{:12} = {:.5e}".format("mupftol", 1e-14) + "\n")

    file.write("    " + "{:12} = {:d}".format("mupfits",   8) + "\n")

    file.write("/\n")
    file.write("\n")

    # numericlist
    file.write("&numericlist\n")
    if nvol == 1:
        file.write("    " + "{:12} = {:d}".format("linitialize", 1) + "\n")
    else:
        file.write("    " + "{:12} = {:d}".format("linitialize", 0) + "\n")
    file.write("    " + "{:12} = {:d}".format("ndiscrete",   2) + "\n")
    file.write("    " + "{:12} = {:d}".format("nquad",      -1) + "\n")
    file.write("    " + "{:12} = {:d}".format("impol",      -4) + "\n")
    file.write("    " + "{:12} = {:d}".format("intor",      -4) + "\n")
    file.write("    " + "{:12} = {:d}".format("lsparse",     0) + "\n")
    file.write("    " + "{:12} = {:d}".format("lsvdiota",    0) + "\n")
    file.write("    " + "{:12} = {:d}".format("imethod",     3) + "\n")
    file.write("    " + "{:12} = {:d}".format("iorder",      2) + "\n")
    file.write("    " + "{:12} = {:d}".format("iprecon",     0) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("iotatol", -1.0) + "\n")
    file.write("/\n")
    file.write("\n")

    # locallist
    file.write("&locallist\n")
    file.write("    " + "{:12} = {:d}".format("lbeltrami", 4) + "\n")
    file.write("    " + "{:12} = {:d}".format("linitgues", 1) + "\n")
    file.write("/\n")
    file.write("\n")

    # globallist
    file.write("&globallist\n")
    file.write("    " + "{:12} = {:d}".format("lfindzero", 1) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("escale", 0.0) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("pcondense", 2.0) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("forcetol", 1e-10) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("c05xtol", 1e-12) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("c05factor", 1e-2) + "\n")
    file.write("    " + "{:12} = .true.".format("lreadgf") + "\n")
    file.write("    " + "{:12} = {:.5e}".format("opsilon", 1.0) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("epsilon", 0.0) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("upsilon", 1.0) + "\n")
    file.write("/\n")
    file.write("\n")

    # diagnosticslist
    file.write("&diagnosticslist\n")
    file.write("    " + "{:12} = {:.5e}".format("odetol", 1e-7) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("absreq", 1e-8) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("relreq", 1e-8) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("absacc", 1e-4) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("epsr",   1e-8) + "\n")
    file.write("    " + "{:12} = {:d}".format("nppts", 400) + "\n")
    file.write("    " + "{:12} = {:d}".format("nptjs", -1) + "\n")
    file.write("    " + "{:12} = .false.".format("lhevalues") + "\n")
    file.write("    " + "{:12} = .false.".format("lhevectors") + "\n")
    file.write("/\n")
    file.write("\n")

    # screenlist
    file.write("&screenlist\n")
    file.write("    " + "{:12} = .true.".format("wpp00aa") + "\n")
    file.write("/\n")
    file.write("\n")

    if nvol != 1:
        for j in range(len(datas["im"])):
            file.write("    " + "{:<5d}".format(int(datas["im"][j])) + "{:<5d}".format(int(datas["in"][j])))
            for i in range(1, ninterface):
                file.write("{:.5e}".format(float(datas["interface_rc"][i, j])) + "{:5}".format("") +
                "{:.5e}".format(float(datas["interface_zs"][i, j])) + "{:5}".format("") +
                "{:.5e}".format(0) + "{:5}".format("") +"{:.5e}".format(0) + "{:5}".format(""))
            file.write("\n")

    file.close()

    return


if __name__ == "__main__":
    pass
