"""Normalize only explicit opt-in refinement additions for legacy evidence."""
import ast


class RewardOnlyV168Registration(ast.NodeTransformer):
    """Remove exactly the reviewed V168 default-off reward and registration additions."""

    names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V168_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V168_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V169_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V169_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V170_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V170_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V171_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V171_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V172_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V172_PROFILES'}

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id in self.names for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        if node.name == 'MjxFixedPlaneV168RewardConfig':
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v168_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v169_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v170_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v171_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu1_v172_stages'):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if any(isinstance(x, ast.Constant) and isinstance(x.value, str)
               and x.value.startswith('fixed_plane_v168_') for x in ast.walk(node.test)):
            return None
        if (isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
                and isinstance(node.test.comparators[0], ast.Name)
                and node.test.comparators[0].id in self.names):
            return None
        return self.generic_visit(node)

    def visit_Dict(self, node):
        pairs = [(k, v) for k, v in zip(node.keys, node.values)
                 if not (isinstance(k, ast.Name) and k.id in self.names)]
        node.keys = [k for k, v in pairs]
        node.values = [v for k, v in pairs]
        return self.generic_visit(node)

    def visit_Tuple(self, node):
        node.elts = [x for x in node.elts if not (
            isinstance(x, ast.Starred) and isinstance(x.value, ast.Name)
            and x.value.id in self.names) and not (
            isinstance(x, ast.Constant) and x.value in ('fplc_v168_gpu0_', 'fplc_v169_gpu0_', 'fplc_v170_gpu0_'))]
        return self.generic_visit(node)

    visit_List = visit_Tuple
    visit_Set = visit_Tuple


class RefinementRegistration(ast.NodeTransformer):
    """Strip versioned registrations and default-off reward/gate extensions."""

    def visit_Module(self, node):
        node = RewardOnlyV168Registration().visit(node)
        return self.generic_visit(node)

    names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V156_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V156_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V157_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V157_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V158_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V158_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V159_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V159_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V160_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V160_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V161_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V161_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V162_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V162_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V163_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V163_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V164_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V164_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V165_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V165_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V166_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V166_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V167_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V167_PROFILES'}

    def visit_Assign(self, node):
        if any(isinstance(t,ast.Name) and t.id in self.names | {'center_limit'}
               for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self,node):
        if isinstance(node.target,ast.Name) and node.target.id == 'max_recent_mean_hit_contact_center_dist_m':
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self,node):
        if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu1_v156_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu1_v157_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu1_v158_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu1_v159_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v160_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v161_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v162_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v163_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v164_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v165_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v166_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v167_stages'):
            return None
        if node.name=='scale_signed_height_reward':
            pairs=[(a,d) for a,d in zip(node.args.kwonlyargs,node.args.kw_defaults) if a.arg!='adjacent_linear']
            node.args.kwonlyargs=[a for a,d in pairs];node.args.kw_defaults=[d for a,d in pairs]
        return self.generic_visit(node)

    def visit_ClassDef(self,node):
        if node.name in ('MjxFixedPlaneV156RewardConfig','MjxFixedPlaneV158RewardConfig','MjxFixedPlaneV159RewardConfig','MjxFixedPlaneV164RewardConfig','MjxFixedPlaneV165RewardConfig','MjxFixedPlaneV166RewardConfig','MjxFixedPlaneV167RewardConfig'):
            return None
        return self.generic_visit(node)

    def visit_If(self,node):
        if isinstance(node.test,ast.Name) and node.test.id=='adjacent_linear':return None
        if any(isinstance(x,ast.Constant) and isinstance(x.value,str)
               and x.value.startswith(('fixed_plane_v158_','fixed_plane_v159_','fixed_plane_v164_','fixed_plane_v165_','fixed_plane_v166_','fixed_plane_v167_')) for x in ast.walk(node.test)):
            return self.visit(node.orelse[0]) if node.orelse else None
        new_entry=(isinstance(node.test,ast.Compare) and len(node.test.ops)==1
            and isinstance(node.test.ops[0],ast.Eq)
            and isinstance(node.test.comparators[0],ast.Name)
            and node.test.comparators[0].id in self.names)
        center_gate=any((isinstance(x,ast.Name) and x.id=='center_limit')
            or (isinstance(x,ast.Attribute) and x.attr=='max_recent_mean_hit_contact_center_dist_m')
            for x in ast.walk(node.test))
        if new_entry or center_gate:
            return None
        return self.generic_visit(node)

    def visit_Dict(self,node):
        pairs=[(k,v) for k,v in zip(node.keys,node.values)
               if not (isinstance(k,ast.Name) and k.id in self.names)]
        node.keys=[k for k,v in pairs]
        node.values=[v for k,v in pairs]
        return self.generic_visit(node)

    def visit_Tuple(self,node):
        node.elts=[x for x in node.elts if not (
            isinstance(x,ast.Starred) and isinstance(x.value,ast.Name) and x.value.id in self.names)
            and not (isinstance(x,ast.Constant) and x.value in (
                'fplc_v156_gpu1_', 'fplc_v157_gpu1_', 'fplc_v158_gpu1_', 'fplc_v159_gpu1_', 'fplc_v160_gpu0_', 'fplc_v161_gpu0_', 'fplc_v162_gpu0_', 'fplc_v163_gpu0_', 'fplc_v164_gpu0_', 'fplc_v165_gpu0_', 'fplc_v166_gpu0_', 'fplc_v167_gpu0_'))]
        return self.generic_visit(node)

    visit_List=visit_Tuple
    visit_Set=visit_Tuple

    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id == 'scale_signed_height_reward':
            node.keywords=[k for k in node.keywords if k.arg not in ('adjacent_multiplier','adjacent_linear')]
        return self.generic_visit(node)


def verify_legacy_source(path,expected_sha):
    """Check exact reviewed hashes and recompute default-mode AST equality."""
    import hashlib
    import json
    from pathlib import Path
    from preflight_fixed_plane_v154 import numerical_source
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    root=Path(__file__).resolve().parents[2]/'outputs/rl_sim'
    proofs=sorted(root.glob('v*_evidence_*/legacy_source_compatibility.json'),reverse=True)
    current=sha(path)
    proof=next((json.loads(p.read_text()).get(path.name) for p in proofs
                if json.loads(p.read_text()).get(path.name,{}).get('current_sha256')==current),None)
    if proof is None:
        raise ValueError('Missing reviewed legacy compatibility proof')
    original=Path(proof['original_path'])
    if sha(original)!=expected_sha or sha(path)!=proof['current_sha256']:
        raise ValueError('Legacy refinement source hash mismatch')
    if numerical_source(path)!=numerical_source(original):
        raise ValueError('Refinement changed an existing numerical path')


class SuccessorRegistration(RefinementRegistration):
    """Remove only explicitly registered successors, retaining V156 behavior."""

    names={'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V157_PROFILE',
           'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V157_PROFILES',
           'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V158_PROFILE',
           'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V158_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V159_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V159_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V160_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V160_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V161_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V161_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V162_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V162_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V163_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V163_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V164_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V164_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V165_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V165_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V166_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V166_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V167_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V167_PROFILES'}

    def visit_AnnAssign(self,node):
        return self.generic_visit(node)

    def visit_ClassDef(self,node):
        if node.name in ('MjxFixedPlaneV158RewardConfig','MjxFixedPlaneV159RewardConfig','MjxFixedPlaneV164RewardConfig','MjxFixedPlaneV165RewardConfig','MjxFixedPlaneV166RewardConfig','MjxFixedPlaneV167RewardConfig'):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self,node):
        if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu1_v157_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu1_v158_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu1_v159_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v160_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v161_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v162_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v163_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v164_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v165_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v166_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v167_stages'):
            return None
        if node.name=='scale_signed_height_reward':
            pairs=[(a,d) for a,d in zip(node.args.kwonlyargs,node.args.kw_defaults) if a.arg!='adjacent_linear']
            node.args.kwonlyargs=[a for a,d in pairs];node.args.kw_defaults=[d for a,d in pairs]
        return self.generic_visit(node)

    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id in self.names for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_If(self,node):
        if isinstance(node.test,ast.Name) and node.test.id=='adjacent_linear':return None
        if any(isinstance(x,ast.Constant) and isinstance(x.value,str)
               and x.value.startswith(('fixed_plane_v158_','fixed_plane_v159_','fixed_plane_v164_','fixed_plane_v165_','fixed_plane_v166_','fixed_plane_v167_')) for x in ast.walk(node.test)):
            return self.visit(node.orelse[0]) if node.orelse else None
        if (isinstance(node.test,ast.Compare) and len(node.test.ops)==1
                and isinstance(node.test.ops[0],ast.Eq)
                and isinstance(node.test.comparators[0],ast.Name)
                and node.test.comparators[0].id in self.names):
            return None
        return self.generic_visit(node)

    def visit_Tuple(self,node):
        node.elts=[x for x in node.elts if not (
            isinstance(x,ast.Starred) and isinstance(x.value,ast.Name) and x.value.id in self.names)
            and not (isinstance(x,ast.Constant) and x.value in ('fplc_v157_gpu1_','fplc_v158_gpu1_', 'fplc_v159_gpu1_', 'fplc_v160_gpu0_', 'fplc_v161_gpu0_', 'fplc_v162_gpu0_', 'fplc_v163_gpu0_', 'fplc_v164_gpu0_', 'fplc_v165_gpu0_', 'fplc_v166_gpu0_', 'fplc_v167_gpu0_'))]
        return self.generic_visit(node)

    visit_List=visit_Tuple
    visit_Set=visit_Tuple

    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='scale_signed_height_reward':
            node.keywords=[k for k in node.keywords if k.arg!='adjacent_linear']
        return self.generic_visit(node)

    def visit_ImportFrom(self,node):
        node.names=[x for x in node.names if x.name not in ('MjxFixedPlaneV158RewardConfig','MjxFixedPlaneV159RewardConfig','MjxFixedPlaneV164RewardConfig','MjxFixedPlaneV165RewardConfig','MjxFixedPlaneV166RewardConfig','MjxFixedPlaneV167RewardConfig')]
        return node if node.names else None


def verify_v156_successor_source(path,expected_sha):
    """Recompute AST equality with the exact V156 probe source."""
    import hashlib
    from pathlib import Path
    original=(Path(__file__).resolve().parents[2]/'outputs/rl_sim/'
        'fplc_v156_gpu1_motion_probe_20261008_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim'/path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('V156 frozen source mismatch')
    normalize=lambda p:ast.dump(SuccessorRegistration().visit(ast.parse(p.read_text())),
                              include_attributes=False)
    if normalize(path)!=normalize(original):
        raise ValueError('Successor changes V156 numerical behavior')


class V160Registration(ast.NodeTransformer):
    """Remove solely the reviewed V160 profile registrations from V159 sources."""

    def visit_Module(self, node):
        node = RewardOnlyV168Registration().visit(node)
        return self.generic_visit(node)

    names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V160_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V160_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V161_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V161_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V162_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V162_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V163_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V163_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V164_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V164_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V165_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V165_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V166_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V166_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V167_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V167_PROFILES'}

    def visit_ClassDef(self, node):
        if node.name in ('MjxFixedPlaneV164RewardConfig','MjxFixedPlaneV165RewardConfig','MjxFixedPlaneV166RewardConfig','MjxFixedPlaneV167RewardConfig'): return None
        return self.generic_visit(node)

    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id in self.names for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self,node):
        if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v160_stages','_fixed_plane_limit_cycle_qvel_rmp_gpu0_v161_stages',
                         '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v162_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v163_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v164_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v165_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v166_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v167_stages'):return None
        return self.generic_visit(node)

    def visit_If(self,node):
        if any(isinstance(x, ast.Constant) and isinstance(x.value, str)
               and x.value.startswith(('fixed_plane_v164_','fixed_plane_v165_','fixed_plane_v166_','fixed_plane_v167_')) for x in ast.walk(node.test)):
            return None
        if (isinstance(node.test,ast.Compare) and len(node.test.ops)==1
            and isinstance(node.test.ops[0],ast.Eq)
            and isinstance(node.test.comparators[0],ast.Name)
            and node.test.comparators[0].id in self.names):return None
        return self.generic_visit(node)

    def visit_Dict(self,node):
        pairs=[(k,v) for k,v in zip(node.keys,node.values)
               if not (isinstance(k,ast.Name) and k.id in self.names)]
        node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
        return self.generic_visit(node)

    def visit_Tuple(self,node):
        node.elts=[x for x in node.elts if not (
            isinstance(x,ast.Starred) and isinstance(x.value,ast.Name) and x.value.id in self.names)
            and not (isinstance(x,ast.Constant) and x.value in ('fplc_v160_gpu0_','fplc_v161_gpu0_', 'fplc_v162_gpu0_', 'fplc_v163_gpu0_', 'fplc_v164_gpu0_', 'fplc_v165_gpu0_', 'fplc_v166_gpu0_', 'fplc_v167_gpu0_'))]
        return self.generic_visit(node)

    visit_List=visit_Tuple
    visit_Set=visit_Tuple


def verify_v159_successor_source(path,expected_sha):
    """Verify only V160 registrations changed, retaining frozen V159 behavior."""
    import hashlib
    from pathlib import Path
    original=(Path(__file__).resolve().parents[2]/'outputs/rl_sim/'
        'fplc_v159_gpu1_mean_quality_formal_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim'/path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('V159 frozen source mismatch')
    normalize=lambda p:ast.dump(V160Registration().visit(ast.parse(p.read_text())),include_attributes=False)
    if normalize(path)!=normalize(original):
        raise ValueError('V160 registration changes V159 numerical behavior')


def verify_v160_successor_source(path,expected_sha):
    """Verify exclusively V161 registrations changed from frozen V160."""
    import hashlib
    from pathlib import Path
    original=(Path(__file__).resolve().parents[2]/'outputs/rl_sim/'
        'fplc_v160_gpu0_mean_quality_formal_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim'/path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('V160 frozen source mismatch')
    class Registration(V160Registration):
        names={'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V161_PROFILE','FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V161_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V162_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V162_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V163_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V163_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V164_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V164_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V165_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V165_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V166_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V166_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V167_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V167_PROFILES'}
        def visit_FunctionDef(self,node):
            if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v161_stages','_fixed_plane_limit_cycle_qvel_rmp_gpu0_v162_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v163_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v164_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v165_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v166_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v167_stages'):return None
            return self.generic_visit(node)
        def visit_Tuple(self,node):
            node.elts=[x for x in node.elts if not (isinstance(x,ast.Starred) and isinstance(x.value,ast.Name) and x.value.id in self.names) and not (isinstance(x,ast.Constant) and x.value in ('fplc_v161_gpu0_','fplc_v162_gpu0_', 'fplc_v163_gpu0_', 'fplc_v164_gpu0_', 'fplc_v165_gpu0_', 'fplc_v166_gpu0_', 'fplc_v167_gpu0_'))]
            return self.generic_visit(node)
        visit_List=visit_Tuple
        visit_Set=visit_Tuple
    normalize=lambda p:ast.dump(Registration().visit(ast.parse(p.read_text())),include_attributes=False)
    if normalize(path)!=normalize(original):raise ValueError('V161 changes V160 numerical behavior')


def verify_v161_successor_source(path, expected_sha):
    """Check only V162 registration changed the frozen V161 numerical path."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v161_gpu0_mean_quality_formal_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V161 frozen source mismatch')
    class Registration(V160Registration):
        names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V162_PROFILE',
                 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V162_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V163_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V163_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V164_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V164_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V165_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V165_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V166_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V166_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V167_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V167_PROFILES'}
        def visit_FunctionDef(self, node):
            if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v162_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v163_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v164_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v165_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v166_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v167_stages'): return None
            return self.generic_visit(node)
        def visit_Tuple(self, node):
            node.elts = [x for x in node.elts if not (
                isinstance(x, ast.Starred) and isinstance(x.value, ast.Name) and x.value.id in self.names)
                and not (isinstance(x, ast.Constant) and x.value in ('fplc_v162_gpu0_', 'fplc_v163_gpu0_', 'fplc_v164_gpu0_', 'fplc_v165_gpu0_', 'fplc_v166_gpu0_', 'fplc_v167_gpu0_'))]
            return self.generic_visit(node)
        visit_List = visit_Tuple
        visit_Set = visit_Tuple
    normalize = lambda p: ast.dump(Registration().visit(ast.parse(p.read_text())), include_attributes=False)
    if normalize(path) != normalize(original):
        raise ValueError('V162 changed V161 numerical behavior')


def verify_v162_successor_source(path, expected_sha):
    """Remove only V163 registrations from the exact frozen V162 source."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v162_gpu0_centering_probe_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V162 frozen source mismatch')
    class Registration(V160Registration):
        names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V163_PROFILE',
                 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V163_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V164_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V164_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V165_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V165_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V166_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V166_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V167_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V167_PROFILES'}
        def visit_FunctionDef(self, node):
            if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v163_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v164_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v165_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v166_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v167_stages'): return None
            return self.generic_visit(node)
        def visit_Tuple(self, node):
            node.elts = [x for x in node.elts if not (
                isinstance(x, ast.Starred) and isinstance(x.value, ast.Name) and x.value.id in self.names)
                and not (isinstance(x, ast.Constant) and x.value in ('fplc_v163_gpu0_', 'fplc_v164_gpu0_', 'fplc_v165_gpu0_', 'fplc_v166_gpu0_', 'fplc_v167_gpu0_'))]
            return self.generic_visit(node)
        visit_List = visit_Tuple
        visit_Set = visit_Tuple
    normalize = lambda p: ast.dump(Registration().visit(ast.parse(p.read_text())), include_attributes=False)
    if normalize(path) != normalize(original):
        raise ValueError('V163 changed V162 numerical behavior')


def verify_v163_successor_source(path, expected_sha):
    """Check only default-off V164 rewards and registrations changed V163."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v163_gpu0_centering_probe_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V163 frozen source mismatch')
    class Registration(V160Registration):
        names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V164_PROFILE',
                 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V164_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V165_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V165_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V166_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V166_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V167_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V167_PROFILES'}
        def visit_FunctionDef(self, node):
            if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v164_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v165_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v166_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v167_stages'): return None
            return self.generic_visit(node)
        def visit_Tuple(self, node):
            node.elts = [x for x in node.elts if not (
                isinstance(x, ast.Starred) and isinstance(x.value, ast.Name) and x.value.id in self.names)
                and not (isinstance(x, ast.Constant) and x.value in ('fplc_v164_gpu0_', 'fplc_v165_gpu0_', 'fplc_v166_gpu0_', 'fplc_v167_gpu0_'))]
            return self.generic_visit(node)
        visit_List = visit_Tuple
        visit_Set = visit_Tuple
    normalize = lambda p: ast.dump(Registration().visit(ast.parse(p.read_text())), include_attributes=False)
    if normalize(path) != normalize(original):
        raise ValueError('V164 changed V163 numerical behavior')


def verify_v164_successor_source(path, expected_sha):
    """Retain every V164 numerical hook and strip only opt-in V165 additions."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v164_gpu0_dense_anchor_probe_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V164 frozen source mismatch')
    class Registration(V160Registration):
        names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V165_PROFILE',
                 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V165_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V166_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V166_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V167_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V167_PROFILES'}
        def visit_ClassDef(self, node):
            if node.name in ('MjxFixedPlaneV165RewardConfig','MjxFixedPlaneV166RewardConfig','MjxFixedPlaneV167RewardConfig'): return None
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v165_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v166_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v167_stages'): return None
            return self.generic_visit(node)
        def visit_If(self, node):
            if any(isinstance(x, ast.Constant) and isinstance(x.value, str)
                   and x.value.startswith(('fixed_plane_v165_','fixed_plane_v166_','fixed_plane_v167_')) for x in ast.walk(node.test)):
                return None
            if (isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq)
                    and isinstance(node.test.comparators[0], ast.Name)
                    and node.test.comparators[0].id in self.names):
                return None
            return self.generic_visit(node)
        def visit_Tuple(self, node):
            node.elts = [x for x in node.elts if not (
                isinstance(x, ast.Starred) and isinstance(x.value, ast.Name) and x.value.id in self.names)
                and not (isinstance(x, ast.Constant) and x.value in ('fplc_v165_gpu0_', 'fplc_v166_gpu0_', 'fplc_v167_gpu0_'), 'fplc_v166_gpu0_', 'fplc_v167_gpu0_')]
            return self.generic_visit(node)
        visit_List = visit_Tuple
        visit_Set = visit_Tuple
    normalize = lambda p: ast.dump(Registration().visit(ast.parse(p.read_text())), include_attributes=False)
    if normalize(path) != normalize(original):
        raise ValueError('V165 changed V164 numerical behavior')


def verify_v165_successor_source(path, expected_sha):
    """Preserve every V165 hook; strip only registered default-off V166 additions."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v165_gpu0_reward_balance_probe_20261009_r2/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V165 frozen source mismatch')
    class Registration(V160Registration):
        names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V166_PROFILE',
                 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V166_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V167_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V167_PROFILES'}
        def visit_ClassDef(self,node):
            if node.name in ('MjxFixedPlaneV166RewardConfig','MjxFixedPlaneV167RewardConfig'): return None
            return self.generic_visit(node)
        def visit_FunctionDef(self,node):
            if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v166_stages','_fixed_plane_limit_cycle_qvel_rmp_gpu0_v167_stages'): return None
            return self.generic_visit(node)
        def visit_If(self,node):
            if any(isinstance(x,ast.Constant) and isinstance(x.value,str)
                   and x.value.startswith(('fixed_plane_v166_','fixed_plane_v167_')) for x in ast.walk(node.test)):
                return None
            if (isinstance(node.test,ast.Compare) and len(node.test.ops)==1
                    and isinstance(node.test.ops[0],ast.Eq)
                    and isinstance(node.test.comparators[0],ast.Name)
                    and node.test.comparators[0].id in self.names):return None
            return self.generic_visit(node)
        def visit_Tuple(self,node):
            node.elts=[x for x in node.elts if not (
                isinstance(x,ast.Starred) and isinstance(x.value,ast.Name) and x.value.id in self.names)
                and not (isinstance(x,ast.Constant) and x.value in ('fplc_v166_gpu0_','fplc_v167_gpu0_'))]
            return self.generic_visit(node)
        visit_List=visit_Tuple
        visit_Set=visit_Tuple
    normalize=lambda p:ast.dump(Registration().visit(ast.parse(p.read_text())),include_attributes=False)
    if normalize(path)!=normalize(original):raise ValueError('V166 changed V165 numerical behavior')


def verify_v166_successor_source(path, expected_sha):
    """Preserve every V166 hook; strip only registered default-off V167 additions."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v166_gpu0_ball_center_probe_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V166 frozen source mismatch')
    class Registration(V160Registration):
        names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V167_PROFILE',
                 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V167_PROFILES'}
        def visit_ClassDef(self,node):
            if node.name == 'MjxFixedPlaneV167RewardConfig': return None
            return self.generic_visit(node)
        def visit_FunctionDef(self,node):
            if node.name == '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v167_stages': return None
            return self.generic_visit(node)
        def visit_If(self,node):
            if any(isinstance(x,ast.Constant) and isinstance(x.value,str)
                   and x.value.startswith('fixed_plane_v167_') for x in ast.walk(node.test)):
                return None
            if (isinstance(node.test,ast.Compare) and len(node.test.ops)==1
                    and isinstance(node.test.ops[0],ast.Eq)
                    and isinstance(node.test.comparators[0],ast.Name)
                    and node.test.comparators[0].id in self.names):return None
            return self.generic_visit(node)
        def visit_Tuple(self,node):
            node.elts=[x for x in node.elts if not (
                isinstance(x,ast.Starred) and isinstance(x.value,ast.Name) and x.value.id in self.names)
                and not (isinstance(x,ast.Constant) and x.value == 'fplc_v167_gpu0_')]
            return self.generic_visit(node)
        visit_List=visit_Tuple
        visit_Set=visit_Tuple
    normalize=lambda p:ast.dump(Registration().visit(ast.parse(p.read_text())),include_attributes=False)
    if normalize(path)!=normalize(original):raise ValueError('V167 changed V166 numerical behavior')


def verify_v167_successor_source(path, expected_sha):
    """Recompute exact V167 behavior after only registered V168 additions."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v167_gpu0_tilt_center_probe_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V167 frozen source mismatch')
    normalize = lambda p: ast.dump(RewardOnlyV168Registration().visit(
        ast.parse(p.read_text())), include_attributes=False)
    if normalize(path) != normalize(original):
        raise ValueError('V168 changed V167 numerical behavior')


def verify_v168_successor_source(path, expected_sha):
    """Strip solely V169 registrations while retaining every V168 reward hook."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v168_gpu0_mean_quality_probe_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V168 frozen source mismatch')
    class Registration(RewardOnlyV168Registration):
        names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V169_PROFILE',
                 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V169_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V170_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V170_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V171_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V171_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V172_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V172_PROFILES'}
        def visit_ClassDef(self, node):
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            return None if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v169_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v170_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v171_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu1_v172_stages') else self.generic_visit(node)
        def visit_If(self, node):
            if (isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq)
                    and isinstance(node.test.comparators[0], ast.Name)
                    and node.test.comparators[0].id in self.names):
                return None
            return self.generic_visit(node)
        def visit_Tuple(self, node):
            node.elts = [x for x in node.elts if not (
                isinstance(x, ast.Starred) and isinstance(x.value, ast.Name)
                and x.value.id in self.names) and not (
                isinstance(x, ast.Constant) and x.value in ('fplc_v169_gpu0_', 'fplc_v170_gpu0_'))]
            return self.generic_visit(node)
        visit_List = visit_Tuple
        visit_Set = visit_Tuple
    normalize = lambda p: ast.dump(Registration().visit(ast.parse(p.read_text())),include_attributes=False)
    if normalize(path) != normalize(original):
        raise ValueError('V169 changed V168 numerical behavior')


def verify_v169_successor_source(path, expected_sha):
    """Strip solely V170 registrations while retaining every V169 reward hook."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v169_gpu0_balanced_quality_probe_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V169 frozen source mismatch')
    class Registration(RewardOnlyV168Registration):
        names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V170_PROFILE',
                 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V170_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V171_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V171_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V172_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V172_PROFILES'}
        def visit_ClassDef(self, node):
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            return None if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v170_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu0_v171_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu1_v172_stages') else self.generic_visit(node)
        def visit_If(self, node):
            if (isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq)
                    and isinstance(node.test.comparators[0], ast.Name)
                    and node.test.comparators[0].id in self.names):
                return None
            return self.generic_visit(node)
        def visit_Tuple(self, node):
            node.elts = [x for x in node.elts if not (
                isinstance(x, ast.Starred) and isinstance(x.value, ast.Name)
                and x.value.id in self.names) and not (
                isinstance(x, ast.Constant) and x.value in ('fplc_v170_gpu0_',))]
            return self.generic_visit(node)
        visit_List = visit_Tuple
        visit_Set = visit_Tuple
    normalize = lambda p: ast.dump(Registration().visit(ast.parse(p.read_text())),include_attributes=False)
    if normalize(path) != normalize(original):
        raise ValueError('V170 changed V169 numerical behavior')


def verify_v170_successor_source(path, expected_sha):
    """Ignore only the reviewed V171 gate-profile registrations for V170."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v170_gpu0_corrective_return_probe_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V170 frozen source mismatch')
    class Registration(RewardOnlyV168Registration):
        names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V171_PROFILE',
                 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V171_PROFILES',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V172_PROFILE',
             'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V172_PROFILES'}
        def visit_ClassDef(self, node):
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            return None if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu0_v171_stages', '_fixed_plane_limit_cycle_qvel_rmp_gpu1_v172_stages') else self.generic_visit(node)
        def visit_If(self, node):
            if (isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq)
                    and isinstance(node.test.comparators[0], ast.Name)
                    and node.test.comparators[0].id in self.names):
                return None
            return self.generic_visit(node)
    normalize = lambda p: ast.dump(Registration().visit(ast.parse(p.read_text())), include_attributes=False)
    if normalize(path) != normalize(original):
        raise ValueError('V171 changed V170 numerical behavior')


def verify_v171_successor_source(path, expected_sha):
    """Ignore only the reviewed V171 gate-profile registrations for V171."""
    import hashlib
    from pathlib import Path
    original = (Path(__file__).resolve().parents[2] / 'outputs/rl_sim/'
        'fplc_v171_gpu0_quality_gates_formal_20261009_r1/source_snapshot/'
        'pingpong_controller/tools/rl_sim' / path.name)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('V171 frozen source mismatch')
    class Registration(RewardOnlyV168Registration):
        names = {'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU1_V172_PROFILE',
                 'FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_V172_PROFILES'}
        def visit_ClassDef(self, node):
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            return None if node.name in ('_fixed_plane_limit_cycle_qvel_rmp_gpu1_v172_stages',) else self.generic_visit(node)
        def visit_If(self, node):
            if (isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq)
                    and isinstance(node.test.comparators[0], ast.Name)
                    and node.test.comparators[0].id in self.names):
                return None
            return self.generic_visit(node)
    normalize = lambda p: ast.dump(Registration().visit(ast.parse(p.read_text())), include_attributes=False)
    if normalize(path) != normalize(original):
        raise ValueError('V172 changed V171 numerical behavior')
