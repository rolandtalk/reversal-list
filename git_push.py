#!/usr/bin/env python3
import subprocess
import os

os.chdir('/Users/rolandtalkonmini/reversal-list')

# Git add
subprocess.run(['/usr/bin/git', 'add', '-A'])

# Git commit
result = subprocess.run(['/usr/bin/git', 'commit', '-m', 'Export to real Excel format with % symbol using SheetJS'], 
                        capture_output=True, text=True)
print("Commit output:", result.stdout, result.stderr)

# Git push
result = subprocess.run(['/usr/bin/git', 'push'], capture_output=True, text=True)
print("Push output:", result.stdout, result.stderr)
