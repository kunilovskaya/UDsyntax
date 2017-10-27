#!/usr/bin/python3
deprels = 'nsubj obj iobj ccomp xcomp obl advcl advmod discourse aux cop mark ' \
                'nmod appos nummod acl amod case conj cc fixed flat compound parataxis orphan ' \
                ' root dep acl:relcl flat:name nsubj:pass nummod:gov aux:pass flat:foreign ' \
                'obl:agent nummod:entity'.split()

for dep in sorted(deprels):
    print("('%s', [])," % (dep), end=" ")
