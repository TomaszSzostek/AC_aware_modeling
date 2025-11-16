"""
Generate fragment structure images for Figure 3 Panel B
Creates PNG images of top AC-added fragments for lollipop chart
"""

import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
import sys

def generate_fragment_images(output_dir: Path, top_n: int = 5):
    """Generate structure images for top N AC-added fragments"""
    
    # Load AC-added fragments
    ac_frag_path = Path('results/reverse_QSAR/LogReg/selected_fragments_with_ACflag.csv')
    if not ac_frag_path.exists():
        print(f"Fragment file not found: {ac_frag_path}")
        sys.exit(1)
    
    df = pd.read_csv(ac_frag_path)
    ac_added = df[df['selected_by'] == 'AC_added'].copy()
    
    if len(ac_added) == 0:
        print("No AC-added fragments found")
        sys.exit(1)
    
    print(f"Found {len(ac_added)} AC-added fragments")
    
    # Sort by ac_enrichment and take top N
    top_frags = ac_added.nlargest(top_n, 'ac_enrichment')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate images
    img_size = (300, 300)
    
    for idx, row in top_frags.iterrows():
        frag_smiles = row['fragment_smiles']
        enrichment = row['ac_enrichment']
        
        # Replace attachment points with carbon (not H) for visualization
        display_smiles = frag_smiles.replace('[*]', 'C')
        
        try:
            mol = Chem.MolFromSmiles(display_smiles)
            if mol is None:
                print(f"Warning: could not parse fragment: {frag_smiles}")
                continue
            
            # Special handling for ethyl fragment (CC) - make it bent to show 2 carbons
            if frag_smiles == '[*]CC[*]':
                from rdkit.Chem import AllChem
                # Force 2D coordinate generation for bent structure
                AllChem.Compute2DCoords(mol)
                # Generate image with explicit 2D coords
                img = Draw.MolToImage(mol, size=img_size, kekulize=False)
            else:
                # Generate image (simple, clean)
                img = Draw.MolToImage(mol, size=img_size)
            
            # Save with fragment index
            img_path = output_dir / f"frag_{idx}.png"
            img.save(img_path)
            print(f"Saved: {img_path.name} (enrichment={enrichment:.1f})")
            
        except Exception as e:
            print(f"Error processing {frag_smiles}: {e}")
    
    # Also save fragment metadata for R
    metadata = top_frags[['fragment_smiles', 'ac_enrichment', 'n_cliff_pairs', 'importance']].copy()
    metadata['fragment_id'] = [f"frag_{idx}" for idx in metadata.index]
    
    metadata_path = output_dir / 'fragment_metadata.csv'
    metadata.to_csv(metadata_path, index=False)
    print(f"\nSaved metadata: {metadata_path}")
    print(f"\nTop {top_n} AC-added fragments by enrichment:")
    print(metadata[['fragment_id', 'ac_enrichment', 'n_cliff_pairs']].to_string(index=False))

if __name__ == "__main__":
    output_dir = Path('R_reporting/figures/fragments')
    generate_fragment_images(output_dir, top_n=5)

