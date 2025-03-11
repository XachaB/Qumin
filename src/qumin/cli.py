import logging
import hydra

from .calc_paradigm_entropy import H_command
from .find_patterns import pat_command
from .find_macroclasses import macroclasses_command
from .make_lattice import lattice_command
from .microclass_heatmap import heatmap_command
from .entropy_heatmap import ent_heatmap_command
from .eval import eval_command
from .utils import Metadata

log = logging.getLogger()


@hydra.main(version_base=None, config_path="config", config_name="qumin")
def qumin_command(cfg):
    log.info(cfg)
    md = Metadata(cfg=cfg)

    if (cfg.patterns is None or cfg.action == "patterns") and \
            cfg.action != 'ent_heatmap':
        not_overab = not cfg.pats.overabundant.keep
        not_defect = not cfg.pats.defective
        for_H = cfg.action == "H"
        for_m = cfg.action == "macroclasses"
        assert not_overab or not (for_H or for_m), "For this calculation, overabundant must be False"
        assert not_defect or not for_m, "For this calculation, defective must be False"
        pat_command(cfg, md)

    if cfg.action in ['H', 'macroclasses', 'lattice', 'heatmap']:
        patterns_md = Metadata(path=cfg.patterns) if cfg.patterns else md

    if cfg.action == "H":
        H_command(cfg, md, patterns_md)
    elif cfg.action == "macroclasses":
        macroclasses_command(cfg, md, patterns_md)
    elif cfg.action == "lattice":
        lattice_command(cfg, md, patterns_md)
    elif cfg.action == "heatmap":
        heatmap_command(cfg, md)
    elif cfg.action == "eval":
        eval_command(cfg, md)

    if (cfg.action == "H" and cfg.entropy.heatmap) or cfg.action == 'ent_heatmap':
        ent_heatmap_command(cfg, md)

    md.save_metadata()
