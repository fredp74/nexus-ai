"""Multi-fact continuation using the tested relations/replay trainer."""
import itertools
import random
import sys
from pathlib import Path

import train_mamba_relations as trainer

previous_data = trainer.controlled


def controlled(tokenizer):
    retained = previous_data(tokenizer)
    first = 'Amelia Beatrice Cedric Daria Emil Greta Henrik Ilona Jasper Katia Lionel Nadine Oscar Petra Quentin Sonia'.split()
    last = 'Aalto Berger Caron Delacroix Eriksen Ferrer Girard Holst Iversen Jensen Laurent Marin Moreau Perrin Renard Vidal'.split()
    adjectives = 'Autumn Bronze Crystal Deep Emerald Evening Fair Gentle Grand Indigo Morning Northern Ocean Peaceful Southern Velvet'.split()
    nouns = 'Arch Bay Bell Cloud Coast Creek Dawn Dragon Field Flame Flower Haven Island Mountain River Star'.split()
    places = 'Bremen Cadiz Como Dundee Faro Girona Lille Lucca Nancy Nice Pisa Porto Siena Trento Trier Vigo'.split()
    instruments = ['electric guitar', 'piano', 'violin', 'drums', 'flute', 'saxophone', 'cello', 'bass guitar']
    genres = ['jazz', 'rock', 'folk music', 'classical music', 'blues', 'country music', 'electronic music', 'black metal']
    foods = ['English', 'French', 'Italian', 'Indian', 'Chinese', 'Japanese', 'Spanish', 'Greek']
    prices = ['less than \u00a315', 'less than \u00a325', 'more than \u00a330', 'more than \u00a345']
    pools = {
        'birthplace': [' '.join(p) for p in itertools.product(first, last)],
        'owner': ['MV ' + ' '.join(p) for p in itertools.product(adjectives, nouns)],
        'price': ['The ' + ' '.join(p) for p in itertools.product(adjectives, nouns)],
    }
    result = {s: {f: [] for f in pools} for s in ('train', 'dev', 'test')}
    provenance = trainer.base.bench.digest(Path(__file__))
    rng = random.Random(2721)
    for family, names in pools.items():
        rng.shuffle(names)
        dimensions = (8, 8) if family == 'birthplace' else ((16, 16) if family == 'owner' else (8, 4))
        pairs = {s: [] for s in result}
        for a, b in itertools.product(range(dimensions[0]), range(dimensions[1])):
            residue = (a + b) % 4
            split = 'dev' if residue == 0 else ('test' if residue == 1 else 'train')
            pairs[split].append((a, b))
        for i, name in enumerate(names[:152]):
            split = 'train' if i < 128 else ('dev' if i < 140 else 'test')
            for variant, (a, b) in enumerate(rng.sample(pairs[split], 2)):
                if family == 'birthplace':
                    instrument, genre = instruments[a], genres[b]
                    city = rng.choice(places)
                    facts = [(name, 'BIRTH_PLACE', city), (name, 'INSTRUMENT', instrument), (name, 'GENRE', genre)]
                    text = name + ' was born in ' + city + ' and plays ' + instrument + ' in the ' + genre + ' genre.'
                    if i % 2:
                        fusion = rng.choice([g for g in genres if g != genre])
                        facts = [(name, 'INSTRUMENT', instrument), (name, 'GENRE', genre), (genre, 'MUSIC_FUSION_GENRE', fusion)]
                        text = name + ' plays ' + instrument + ' in the ' + genre + ' genre. ' + fusion.capitalize() + ' is a fusion of ' + genre + '.'
                elif family == 'owner':
                    location, foundation = places[a], places[b]
                    owner = rng.choice(adjectives) + ' Cruises'
                    engine = rng.choice(nouns) + ' Industries'
                    facts = [(name, 'OWNER', owner), (name, 'POWER_TYPE', engine),
                             (engine, 'LOCATION', location), (engine, 'FOUNDATION_PLACE', foundation)]
                    text = name + ' is owned by ' + owner + ' and powered by ' + engine + '. ' + engine + ' was founded in ' + foundation + ' and is located in ' + location + '.'
                else:
                    food, price = foods[a], prices[b]
                    near = 'The ' + rng.choice(adjectives) + ' Cafe'
                    friendly = rng.choice([True, False])
                    facts = [(name, 'eatType', 'restaurant'), (name, 'food', food),
                             (name, 'priceRange', price), (name, 'familyFriendly', 'yes' if friendly else 'no'),
                             (name, 'near', near)]
                    text = name + ' is a restaurant serving ' + food + ' food, priced at ' + price + ', near ' + near + '. It is ' + ('' if friendly else 'not ') + 'family friendly.'
                rng.shuffle(facts)
                prompt = 'Facts:\n' + '\n'.join(' | '.join(t) for t in facts) + '\nText:'
                prefix = tokenizer.encode(prompt)
                answer = tokenizer.encode(' ' + text)
                if tokenizer.encode(prompt + ' ' + text) != prefix + answer:
                    raise ValueError('Token boundary changed')
                ids = prefix + answer + [tokenizer.eos_token_id]
                if len(ids) > 128:
                    raise ValueError('Example exceeds 128 tokens: ' + name)
                result[split][family].append(dict(
                    index=name + ':' + str(variant), entity=name,
                    pair=[a, b], kind='multifact', generator_sha256=provenance,
                    prompt=prompt, reference=text, ids=ids,
                    labels=[-100] * len(prefix) + answer + [tokenizer.eos_token_id]))
    for family in pools:
        result['train'][family].extend(retained['train'][family])
    for split in ('dev', 'test'):
        result[split]['retained'] = [
            row for rows in retained[split].values() for row in rows[:4]]
    print('Prepared:', {s: {f: len(rows) for f, rows in groups.items()}
                         for s, groups in result.items()}, flush=True)
    return result


def main():
    trainer.controlled = controlled
    defaults = {
        '--initial': 'runs/mamba130m_relations_replay/best_adapter',
        '--run': 'runs/mamba130m_multifacts',
        '--steps': '800',
    }
    for flag, value in defaults.items():
        if not any(a == flag or a.startswith(flag + '=') for a in sys.argv[1:]):
            sys.argv.extend([flag, value])
    trainer.main()


if __name__ == '__main__':
    main()
