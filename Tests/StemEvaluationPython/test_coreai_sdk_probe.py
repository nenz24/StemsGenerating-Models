"""Probe reporting and honest failure capture; Apple tools are fixtures, not a device test."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[2]


class CoreAISDKProbeTests(unittest.TestCase):
    def tools(self,base,available):
        sdk=base/'iPhoneOS27.0 SDK.sdk';sdk.mkdir()
        if available:
            framework=sdk/'System/Library/Frameworks/CoreAI.framework'
            for relative,data in {
                'Headers/Fixture.h':'// Public header fixture only\n',
                'Modules/CoreAI.swiftmodule/arm64-apple-ios.swiftinterface':'// Public wrapper fixture only\n@_exported public import CoreAIDelegates\n',
                'Modules/CoreAI.swiftmodule/arm64-apple-ios.private.swiftinterface':'// SPI fixture excluded\n',
                'Modules/CoreAI.swiftmodule/arm64-apple-ios.package.swiftinterface':'// Package fixture excluded\n',
                'Modules/CoreAI.swiftmodule/arm64-apple-ios.swiftmodule':'compiled fixture excluded',
                'PrivateHeaders/Secret.h':'private fixture excluded',
            }.items():
                p=framework/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
        binary=base/'bin';binary.mkdir()
        scripts={
            'xcrun':'''import json,os,sys
from pathlib import Path
a=sys.argv[1:]
if a==['--sdk','iphoneos','--show-sdk-path']:print(os.environ['PROBE_FAKE_SDK'])
elif 'swiftc' in a:
 assert a[a.index('-target')+1]=='arm64-apple-ios27.0'
 assert ('-typecheck' in a or '-scan-dependencies' in a) and Path(a[-1]).read_text()=='import CoreAI\\n'
 cache=Path(a[a.index('-module-cache-path')+1]);(cache/'compiled.fixture').write_text('excluded')
 if os.environ['PROBE_IMPORT_AVAILABLE']!='1':
  print('error: no such module CoreAI',file=sys.stderr);sys.exit(1)
 if '-scan-dependencies' in a:
  assert a[a.index('-module-name')+1]=='CoreAIProbe'
  if os.environ.get('PROBE_SCAN_FAILURE')=='1':
   print('fixture: dependency scanning failed',file=sys.stderr);sys.exit(7)
  print(json.dumps({'fixture_only':True,'modules':[{'swift':'CoreAI'},{'swift':'CoreAIDelegates'}]}))
else:raise AssertionError('unexpected xcrun command: '+str(a))
''',
            'xcodebuild':'''import sys
if sys.argv[1:]==['-version']:print('Xcode 27.0\\nBuild version 27A266a')
elif sys.argv[1:]==['-showsdks']:print('iOS 27.0 -sdk iphoneos27.0')
else:raise AssertionError('unexpected xcodebuild command')
''',
            'ditto':'''import sys,zipfile
from pathlib import Path
source=Path(sys.argv[-2]);target=Path(sys.argv[-1])
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
 for p in source.rglob('*'):
  if p.is_file():archive.write(p,source.name+'/'+str(p.relative_to(source)))
'''}
        for name,source in scripts.items():
            p=binary/name;p.write_text('#!'+sys.executable+'\n'+source);p.chmod(0o755)
        temporary=base/'temporary';temporary.mkdir()
        env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],PROBE_FAKE_SDK=str(sdk),
                 PROBE_IMPORT_AVAILABLE='1' if available else '0',TMPDIR=str(temporary))
        return env

    def run_probe(self,base,env):
        out=base/'public reports'
        result=subprocess.run(['bash',str(ROOT/'scripts/probe_coreai_sdk.sh'),str(out)],env=env,text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result,sorted(out.glob('*.zip'))

    def test_public_interfaces_and_import_result_are_reported_without_private_or_binary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);env=self.tools(base,True);result,archives=self.run_probe(base,env)
            self.assertIn('succeeded (type-check only)',result.stdout)
            self.assertEqual(len(archives),1);original=archives[0].read_bytes()
            with zipfile.ZipFile(archives[0]) as z:
                self.assertIsNone(z.testzip());names=z.namelist()
                self.assertTrue(any(n.endswith('/Fixture.h') for n in names))
                self.assertTrue(any(n.endswith('.swiftinterface') for n in names))
                self.assertFalse(any('PrivateHeaders' in n or n.endswith('.swiftmodule') or n.endswith('compiled.fixture') for n in names))
                self.assertFalse(any(n.endswith('.private.swiftinterface') or n.endswith('.package.swiftinterface') for n in names))
                readme=z.read(next(n for n in names if n.endswith('/README.txt'))).decode()
                self.assertIn('No checkpoint was loaded',readme)
                dependencies=z.read(next(n for n in names if n.endswith('/dependencies.json'))).decode()
                self.assertIn('CoreAIDelegates',dependencies)
            _,again=self.run_probe(base,env)
            self.assertEqual(len(again),2);self.assertEqual(archives[0].read_bytes(),original)

    def test_missing_module_is_a_recorded_failure_with_no_substitute_framework(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);env=self.tools(base,False);result,archives=self.run_probe(base,env)
            self.assertIn('CoreAI import: failed',result.stdout)
            with zipfile.ZipFile(archives[0]) as z:
                names=z.namelist()
                log=z.read(next(n for n in names if n.endswith('/import-log.txt'))).decode()
                self.assertIn('no such module CoreAI',log)
                self.assertFalse(any('/interfaces/' in n for n in names))
                self.assertEqual(z.read(next(n for n in names if n.endswith('/ImportProbe.swift'))),b'import CoreAI\n')

    def test_wrapper_reexport_collects_public_delegate_module_or_reports_it_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);env=self.tools(base,True)
            result,archives=self.run_probe(base,env)
            self.assertIn('CoreAIDelegates public source files: 0',result.stdout)
            with zipfile.ZipFile(archives[0]) as z:
                locations=z.read(next(n for n in z.namelist() if n.endswith('/interface-locations.txt'))).decode()
                self.assertIn('Not found:',locations)
                self.assertIn('CoreAIDelegates.framework',locations)
            sdk=Path(env['PROBE_FAKE_SDK'])
            delegate=sdk/'usr/lib/swift/CoreAIDelegates.swiftmodule/arm64-apple-ios.swiftinterface'
            delegate.parent.mkdir(parents=True)
            delegate.write_text('// Synthetic public dependency fixture only\npublic enum FixtureType {}\n')
            result,archives=self.run_probe(base,env)
            self.assertIn('CoreAIDelegates public source files: 1',result.stdout)
            for archive in archives:
                with zipfile.ZipFile(archive) as z:
                    counts=z.read(next(n for n in z.namelist() if n.endswith('/interface-counts.txt'))).decode()
                    if 'CoreAIDelegates public source files: 1' in counts:
                        self.assertTrue(any('/CoreAIDelegates/sdk-swift-overlay/' in n for n in z.namelist()))
                        break
            else:
                self.fail('No report collected the public delegate interface')

    def test_scan_failure_is_reported_even_when_module_import_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);env=self.tools(base,True);env['PROBE_SCAN_FAILURE']='1'
            result,archives=self.run_probe(base,env)
            self.assertIn('CoreAI import: succeeded',result.stdout)
            self.assertIn('Dependency scan: failed; compiler exit 7',result.stdout)
            with zipfile.ZipFile(archives[0]) as z:
                log=z.read(next(n for n in z.namelist() if n.endswith('/dependency-scan-log.txt'))).decode()
                self.assertIn('fixture: dependency scanning failed',log)

    def test_subframework_dependencies_are_collected_through_sdk_directory_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);env=self.tools(base,True);sdk=Path(env['PROBE_FAKE_SDK'])
            backing=base/'SDK subframework backing';backing.mkdir()
            (sdk/'System/Library/SubFrameworks').symlink_to(backing,target_is_directory=True)
            modules=('CoreAIDelegates','CoreAIRuntime','CoreAICompiler','CoreAICommon','CoreAIAsset')
            for module in modules:
                source=backing/(module+'.framework')/'Modules'/(module+'.swiftmodule')/'arm64e-apple-ios.swiftinterface'
                source.parent.mkdir(parents=True)
                source.write_text('// Synthetic dependency fixture only\npublic enum FixtureType {}\n')
                source.with_name('arm64e-apple-ios.private.swiftinterface').write_text('// Excluded SPI fixture\n')
            result,archives=self.run_probe(base,env)
            with zipfile.ZipFile(archives[0]) as z:
                names=z.namelist()
                for module in modules:
                    self.assertIn(module+' public source files: 1',result.stdout)
                    self.assertTrue(any('/'+module+'/SubFrameworks/modules/' in n for n in names))
                self.assertFalse(any(n.endswith('.private.swiftinterface') for n in names))


if __name__=='__main__':unittest.main()
