#!/usr/bin/env python3
import subprocess
import os

os.chdir('/Users/rolandtalkonmini/reversal-list')

# Git add
subprocess.run(['git', 'add', '-A'])

# Git commit
result = subprocess.run(['git', 'commit', '-m', 'Fix CSV export format - remove % suffix for mobile Excel compatibility'], 
                        capture_output=True, text=True)
print("Commit output:", result.stdout, result.stderr)

# Git push
result = subprocess.run(['git', 'push'], capture_output=True, text=True)
print("Push output:", result.stdout, result.stderr)
