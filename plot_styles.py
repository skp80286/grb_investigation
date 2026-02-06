freq_dict = {
    "r": 481130569731985.2,
    "g": 628495719077568.1,
    "i": 393170436721311.5,
    "z": 328215960148894.2,
    "J": 2.4e14,
    "6_GHz": 6e9,
    "10_GHz": 10e9,
    "swift": 2.18e18,
    "swift_1keV": 2.42e17,
}

labels_dict = {
    "6_GHz": r"6 GHz",
    "10_GHz": r"10 GHz",
    "swift": r"10 keV",
    "swift_1keV": r"1 keV",
    "g": r"g",
    "r": r"r",
    "i": r"i",
    "z": r"z",
    "J": r"J",
}

shifts = {
    "6_GHz": 300,
    "10_GHz": 100,
    "swift": 1,
    "swift_1keV": 1,
    "g": 1 / 4,
    "r": 1,
    "i": 4,
    "z": 12,
    "J": 32,
}

color_dict = {
    "6_GHz": "magenta",
    "10_GHz": "royalblue",
    "swift": "tab:purple",
    "swift_1keV": "mediumslateblue",
    "g": "darkgreen",
    "r": "tab:red",
    "i": "goldenrod",
    "z": "peru",
    "J": "olive",
}

chain_color_dict = {
    "6_GHz": "plum",
    "10_GHz": "skyblue",
    "swift": "thistle",
    "swift_1keV": "lavender",
    "g": "mediumaquamarine",
    "r": "coral",
    "i": "khaki",
    "z": "peachpuff",
    "J": "tan",
}

freq_dict = {
    "r": 4.81e14,
    "i": 3.93e14,
    "o": 3.93e14,
    "XRT": 2.18e18,
    "X": 10e9,
    "ATCA": 16.7e9,
    "ATCA_21": 21.2e9,
    "GMRT": 1.26e9,
    # "S": 3.1e9,
}

labels_dict = {
    "r": r"r",
    "i": r"i",
    "o": r"o",
    "XRT": r"10 keV",
    "X": r"10 GHz",
    "ATCA": r"16.7 GHz",
    "ATCA_21": r"21.2 GHz",
    "GMRT": r"1.26 GHz",
    "S": r"3.1 GHz",
}

shifts = {
    "r": 1,
    "i": 3,
    "o": 2,
    "XRT": 100,
    "X": 50,
    "ATCA": 100,
    "ATCA_21": 200,
    "GMRT": 10,
    "S": 20,
}

color_dict = {
    "r": "tab:red",
    "i": "darkgoldenrod",
    "o": "tab:orange",
    "XRT": "tab:purple",
    "X": "teal",
    "ATCA": "magenta",
    "ATCA_21": "darkmagenta",
    "GMRT": "royalblue",
    "S": "deepskyblue",
}
chain_color_dict = {
    "r": "lightcoral",
    "i": "khaki",
    "o": "peachpuff",
    "XRT": "lavender",
    "X": "mediumturquoise",
    "ATCA": "plum",
    "ATCA_21": "thistle",
    "GMRT": "lightsteelblue",
    "S": "lightblue",
}

plt.style.use(['science', 'high-vis'])

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 5,  # minimum allowed by Nature
    "axes.titlesize": 6,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,  # embed fonts as TrueType
    "ps.fonttype": 42,
    # "figure.dpi": 300,  # ensure high-res bitmap export when needed
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.75,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,

})