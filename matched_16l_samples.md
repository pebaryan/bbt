# Matched 16L Samples

Sampling: temp=0.85, top_k=40, max_new=220, seeds=[99]

> Note: the 82M GA dim=16 checkpoint was overwritten by the resumed 1B run; GA dim8 @82M is included as the preserved GA 82M reference.

## Vanilla 16L @82M

- checkpoint: `/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla_82M_backup.pt`
- step: `5000`
- tokens: `81920000`
- clean_ce: `1.221086859703064`
- block_ce: `2.0328173637390137`

### prompt='Once upon a time', seed=99

```text
Once upon a time, there was a good bird dog named Anna. Lucy learned that her mom said help her mom her goal and the radio. It had done learned by a walking to keep the bird and the wolf count. But when she saw a cat nodded in the floor
```

### prompt='The meaning of life is', seed=99

```text
The meaning of life is sometimes and qualifying similar for disure feelings . The companies were classic does government and and treatment standard to standals shall . The probably was happy to move than when I thought can work his can meltat
```

### prompt='In a shocking finding, scientists discovered that', seed=99

```text
In a shocking finding, scientists discovered that sometimes that they were continued to be colleged for the late afterward, charges does government and and treatment standard . 
 Brown the APLEAPD Controducts in his way completes with Novie and Point of China Chamble o
```

## GA 16L dim8 @82M

- checkpoint: `/home/peb/data/bbt_checkpoints/blockwise_16L_ga.pt`
- step: `5000`
- tokens: `81920000`
- clean_ce: `1.2600730657577515`
- block_ce: `0.8923566937446594`

### prompt='Once upon a time', seed=99

```text
Once upon a time, there was a good bird named Danty. So, Whip Tim liked to play with his friends.
One day, Sam and a raan at the park to play with it all day. He wanted to eat it his wagon for his wand and  saw one berries. Fluffy flew 
```

### prompt='The meaning of life is', seed=99

```text
The meaning of life is sometimes as one of the states. And adbustipangly fundamental wifittines, chaftings encouragey expressantation by distances of being allegated ouh non-fug economic problem dotations and ships. I would other receive life
```

### prompt='In a shocking finding, scientists discovered that', seed=99

```text
In a shocking finding, scientists discovered that sometimes a soque hafpanes, as he necdived plans and the couch afterfuly pehs gettine again and the raan at the gas stalh shoking and feeling commentations in caching at compensions and which we economorateking at furth
```

## Vanilla 16L @1B

- checkpoint: `/home/peb/data/bbt_checkpoints/blockwise_16L_vanilla.pt`
- step: `61036`
- tokens: `1000013824`
- clean_ce: `0.9871664047241211`
- block_ce: `0.3567933440208435`

### prompt='Once upon a time', seed=99

```text
Once upon a time, there was a girl named Lily. She had a unique day at the park. Lily was scared of her girl and didn't know what to do. 
Suddenly, her mom came over and saw Lily hiding under her bed. She took out a big dog and showed h
```

### prompt='The meaning of life is', seed=99

```text
The meaning of life is sometimes a source of basic mom they disagree in an international bow even when indicating the expertise. Organizational conservations are also aware that even inheritance does not lose which we enable them to share.
Th
```

### prompt='In a shocking finding, scientists discovered that', seed=99

```text
In a shocking finding, scientists discovered that sometimes the quality of children have luckily launched papers and crystal have indicated about the radioactive gas stale shocking.
One former radioactive economic product details was a growing officer discovered in the
```

## GA 16L dim16 @1B

- checkpoint: `/home/peb/data/bbt_checkpoints/blockwise_16L_ga_dim16.pt`
- step: `61036`
- tokens: `1000013824`
- clean_ce: `0.8068149089813232`
- block_ce: `0.6466562151908875`

### prompt='Once upon a time', seed=99

```text
Once upon a time, there was a little girl named Mia. She had a glass of candy and lots of people. One day, she found a shiny toy and she wanted to play with it. She thought it was too attractive. But she had a bow and obediently filled 
```

### prompt='The meaning of life is', seed=99

```text
The meaning of life is something that they feel common they didn’t get burned. We’ve been seeching the emotion of the meaning and that you’re considering the means not, but it can be water down.
Real Natural and Point of China Champions
```

### prompt='In a shocking finding, scientists discovered that', seed=99

```text
In a shocking finding, scientists discovered that they might be a lohite and they they discovered a fun adventure. Lily and Sam were her grandma there, and they both smiled and became friends.␀Once upon a time, there was a little boy named Timmy. Timmy loved to play wi
```
