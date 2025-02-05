import os
import json
import glob

path = 'data/BaileyLabels/BaileyLabelSet'

files = glob.glob(path + '/*.jpg')
files = [os.path.basename(file) for file in files]

# string processing
metadata = {file: {} for file in files}

synonyms = {
    'compta': 'compta', 
    'comta': 'compta', 
    'dissiliodinium': 'dissiliodinium', 
    'dissilodinium': 'dissiliodinium',
    'rigaudella': 'rigaudella', 
    'rigfaudella': 'rigaudella',
    'sueculosphaeridium': 'surculosphaeridium', 
    'surcdulosphaeridium': 'surculosphaeridium', 
    'surculosphaeridium': 'surculosphaeridium',
    'surculospharidium': 'surculosphaeridium',
    'tebua': 'tenua',
    'tenua': 'tenua',
    'vestitum': 'vestitum', 
    'vestiyum': 'vestitum',
    'yenua': 'tenua',
}

for i, file in enumerate(files):
    key = file
    # extract well id markers in strings c1, c2, c3, c4
    file = file.replace('crop__', '')
    c1 = file.split('_')[0]
    file = file.replace(c1 + '_', '')
    c2 = file.split('-')[0]
    file = file.replace(c2 + '-', '')
    c3 = file.split('_')[0]
    file = file.replace(c3 + '_', '')
    c4 = file.split('_')[0]
    try:
        c4 = float(c4)
        c4 = ''
    except:
        if c4 == '368mDC':
            c4 = ''
        else:
            c4 = c4
            file = file.replace(c4 + '_', '')
    well_id = c1 + '_' + c2 + '_' + c3 + '_' + c4
    
    metadata[key]['well_id'] = well_id
    
    # extract depth
    d = file.split('_')[0]
    d = d.replace('mDC', '')
    file = file.replace(d, '')
    
    metadata[key]['depth'] = d
    
    # remove various markers
    r = file.split('C')[0]
    file = file.replace(r, '')
    file = file.replace('C', '')
    file = file.lstrip('_')
    r = file.split('B')[0]
    file = file.replace(r, '')
    file = file.lstrip('B')
    for _ in range(3):
        n = file[0]
        try:
            n = int(n)
            file = file[1:]
        except:
            pass
    file = file.lstrip('_')
    
    # extract genus and species
    genus = file.split('_')[0]
    file = file.replace(genus + '_', '')
    file = file.replace('.jpg', '')
    genus = genus.lower()
    
    try:
        genus = synonyms[genus]
    except:
        pass
    
    metadata[key]['genus'] = genus
    
    species = file.split('_')[0]
    #print(file)
    species = species.split('-')[0]
    # remove whitespace
    species = species.replace(' ', '')
    species = species.lower()
    
    try:
        species = synonyms[species]
    except:
        pass

    metadata[key]['species'] = species
    
# save metadata
with open('data/BaileyLabels/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
