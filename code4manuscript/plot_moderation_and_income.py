"""Redraw Figures 4, 5 and S7–S11 from saved plot-ready estimates."""
import argparse
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
import fit_contextual_moderation as moderation
import fit_income_flow_strata as income


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', choices=['4', '5', 'S7', 'S8', 'S9', 'S10', 'S11'])
    figure = parser.parse_args().figure
    data = PROJECT_ROOT / 'data4results'
    if figure in {'4', 'S7', 'S8'}:
        if figure == '4':
            source = data / '01_Main_and_extended_figures/Figure_4/Fig4_DynamicModeration_pair.csv'
            items, columns, legend = moderation.MAIN_MODS, 4, True
        elif figure == 'S7':
            source = data / '02_Supplementary_figures/Figure_S07/FigS4a_Results4_SupplementaryModerators_pair.csv'
            items, columns, legend = moderation.SUPP_MODS[:3], 3, True
        else:
            source = data / '02_Supplementary_figures/Figure_S08/FigS4b_Results4_SupplementaryModerators_pair.csv'
            items, columns, legend = moderation.SUPP_MODS[3:], 3, True
        (moderation.OUT / 'data').mkdir(parents=True, exist_ok=True)
        curves = pd.read_csv(source)
        moderation.plot_grid(curves, moderation.POLICIES, [key for key, _ in items],
                             dict(items), f'Figure_{figure}', columns, legend)
    else:
        files = {
            '5': ('01_Main_and_extended_figures/Figure_5/Fig5_EconomicComposition_quintile_pair.csv', income.PRIMARY_OUTCOMES, 'all_available'),
            'S9': ('02_Supplementary_figures/Figure_S09/FigS5_EconomicCompositionFlow_quintile_pair.csv', income.SUPPLEMENT_OUTCOMES, 'all_available'),
            'S10': ('02_Supplementary_figures/Figure_S10/FigS6_EconomicComposition_quintile_excluding_2022_2023_pair.csv', income.PRIMARY_OUTCOMES, 'exclude_2022_2023'),
            'S11': ('02_Supplementary_figures/Figure_S11/FigS7_EconomicCompositionFlow_quintile_excluding_2022_2023_pair.csv', income.SUPPLEMENT_OUTCOMES, 'exclude_2022_2023'),
        }
        source, outcomes, sample = files[figure]
        income.plot_figure(pd.read_csv(data / source), outcomes, f'Figure_{figure}', sample)


if __name__ == '__main__':
    main()
