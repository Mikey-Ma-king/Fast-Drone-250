#!/usr/bin/python3
"""Inventory today's logs and archive selected completed landings, read-only.

Success selection follows operator context (one known failed docking run),
corroborated by takeoff -> descent -> stop, landed/disarm events. A stop command
alone is NOT proof of physical landing success. No tracking-error threshold or
ground-truth alignment is used to label success. Original logs are never edited.
"""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import rosbag

WORKSPACE = Path(__file__).resolve().parents[1]
LOG_ROOT = Path('/home/pc/perching_bag')
OUT = WORKSPACE / 'analysis/exports'
DATE = '20261120'
ZONE = dt.timezone(dt.timedelta(hours=8))
SELECTED = {
    '20261120_175535_vicon_afbd89',
    '20261120_181526_vicon_05b7fc',
    '20261120_183726_vicon_0f0610',
    '20261120_184938_vicon_22feb6',
}
FAILURE = '20261120_171624_vicon_a554af'
TOPICS = ['/mode_manager','/px4ctrl/takeoff_land','/mavros/state','/mavros/extended_state']


def clock(t):
    return dt.datetime.fromtimestamp(t, ZONE).isoformat(timespec='milliseconds')


def inventory(folder):
    run = json.loads((folder/'run.json').read_text())
    row = dict(run=folder.name, source_directory=str(folder), selected=folder.name in SELECTED,
               recorder_state=run.get('state'), bag_files=[], active_files=[], events=[])
    for p in sorted(folder.glob('*.bag')):
        with rosbag.Bag(str(p)) as b:
            row['bag_files'].append(dict(name=p.name,bytes=p.stat().st_size,
                                        start=clock(b.get_start_time()),end=clock(b.get_end_time()),
                                        duration_s=b.get_end_time()-b.get_start_time()))
            previous={}
            for topic,m,stamp in b.read_messages(topics=TOPICS):
                if topic=='/mode_manager':value=int(m.pose.orientation.w)
                elif topic=='/px4ctrl/takeoff_land':value=int(m.takeoff_land_cmd)
                elif topic=='/mavros/state':value=dict(mode=m.mode,armed=bool(m.armed))
                else:value=dict(landed_state=m.landed_state,vtol_state=m.vtol_state)
                if topic in TOPICS[:2] or previous.get(topic)!=value:
                    row['events'].append(dict(topic=topic,time=clock(stamp.to_sec()),value=value))
                previous[topic]=value
    row['active_files']=[dict(name=p.name,bytes=p.stat().st_size) for p in sorted(folder.glob('*.bag.active'))]
    if folder.name==FAILURE:
        row['category']='excluded_known_failed_landing'
        row['reason']='此前已确认未对准平台的失败降落（含里程计中断）；自动停桨不能当作成功证据。'
    elif row['selected']:
        row['category']='included_success_from_operator_context_with_landing_sequence'
        row['reason']='按操作者今日仅一次失败的说明纳入；日志具有自动起飞、下降、停桨及之后着陆/解除解锁记录。'
    elif row['active_files']:
        row['category']='excluded_unfinalized_recording'
        row['reason']='录制未正常关闭；原 .bag.active 不作修改。副本恢复检查另记，不按失败降落计数。'
    else:
        row['category']='excluded_no_completed_automatic_docking_sequence'
        row['reason']='调试、未飞行、接管或未完整录到自动降落过程；不计为失败降落。'
    if row['selected']:
        assert row['recorder_state']=='closed' and row['bag_files'] and not row['active_files'],row
        assert any(e['topic']=='/px4ctrl/takeoff_land' and e['value']==1 for e in row['events']),row
        assert any(e['topic']=='/px4ctrl/takeoff_land' and e['value']==2 for e in row['events']),row
        assert any(e['topic']=='/mode_manager' and e['value']==2 for e in row['events']),row
        assert any(e['topic']=='/mode_manager' and e['value']==-1 for e in row['events']),row
        stop=max(e['time'] for e in row['events'] if e['topic']=='/px4ctrl/takeoff_land' and e['value']==2)
        assert any(e['topic']=='/mavros/state' and not e['value']['armed'] and e['time']>stop for e in row['events']),row
        assert any(e['topic']=='/mavros/extended_state' and e['value']['landed_state']==1 and e['time']>stop for e in row['events']),row
    row['bag_bytes']=sum(v['bytes'] for v in row['bag_files'])
    row['bag_duration_s']=sum(v['duration_s'] for v in row['bag_files'])
    return row


def inspect_active_copy(folder, row):
    """Recover only a temporary copy to check for an overlooked landing."""
    with tempfile.TemporaryDirectory(prefix='quaddock_active_inventory_') as temporary:
        for original in sorted(folder.glob('*.bag.active')):
            source_stat=original.stat()
            copy=Path(temporary)/'recovered_copy.bag'
            shutil.copy2(str(original),str(copy))
            try:
                with rosbag.Bag(str(copy),'a',allow_unindexed=True) as b:
                    for _ in b.reindex():pass
                events=[]
                with rosbag.Bag(str(copy)) as b:
                    for topic,m,stamp in b.read_messages(topics=TOPICS[:2]):
                        events.append(dict(topic=topic,time=clock(stamp.to_sec()),
                                           value=int(m.pose.orientation.w) if topic==TOPICS[0] else int(m.takeoff_land_cmd)))
                row['active_copy_check']=dict(original_unchanged=original.stat().st_size==source_stat.st_size and original.stat().st_mtime_ns==source_stat.st_mtime_ns,
                                               recovered_control_events=events)
                assert not any(e['topic']=='/px4ctrl/takeoff_land' and e['value']==2 for e in events), 'Unreviewed landing in active copy'
            except rosbag.ROSBagException as error:
                row['active_copy_check']=dict(error=str(error),result='unverified; excluded')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    archive=OUT/(DATE+'_成功降落数据.zip')
    assert not archive.exists(),str(archive)+' already exists; refusing overwrite'
    inventory_rows=[]
    for folder in sorted(LOG_ROOT.glob(DATE+'_*')):
        if not folder.is_dir():continue
        row=inventory(folder)
        if row['active_files']:inspect_active_copy(folder,row)
        inventory_rows.append(row)
        print('INVENTORY',folder.name,row['category'],flush=True)
    assert {r['run'] for r in inventory_rows if r['selected']}==SELECTED
    manifest=dict(date='2026-11-20',timezone='Asia/Shanghai',source_root=str(LOG_ROOT),
                  included_runs=sorted(SELECTED),excluded_known_failure=FAILURE,
                  selection_basis=__doc__,
                  original_logs_preserved=True,run_inventory=inventory_rows)
    manifest_path=OUT/(DATE+'_打包清单.json')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    csv_path=OUT/(DATE+'_打包清单.csv')
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['运行目录','纳入压缩包','分类','原始日志目录','bag数量','bag总字节','记录总时长秒','说明'])
        for r in inventory_rows:
            writer.writerow([r['run'],'是' if r['selected'] else '否',r['category'],r['source_directory'],
                             len(r['bag_files']),r['bag_bytes'],round(r['bag_duration_s'],3),r['reason']])
    hashes={};fingerprints={}
    partial=archive.with_suffix('.zip.partial')
    assert not partial.exists(),str(partial)+' already exists; refusing overwrite'
    with zipfile.ZipFile(str(partial),'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name in sorted(SELECTED):
            for source in sorted((LOG_ROOT/name).rglob('*')):
                if not source.is_file():continue
                assert not source.is_symlink(),str(source)
                member='successful_landings_20261120/'+str(source.relative_to(LOG_ROOT))
                before=source.stat();fingerprints[str(source)]=(before.st_size,before.st_mtime_ns)
                digest=hashlib.sha256()
                info=zipfile.ZipInfo.from_file(str(source),arcname=member)
                info.compress_type=zipfile.ZIP_DEFLATED
                info._compresslevel=6
                with source.open('rb') as src,z.open(info,'w',force_zip64=True) as dst:
                    while True:
                        chunk=src.read(1024*1024)
                        if not chunk:break
                        digest.update(chunk);dst.write(chunk)
                after=source.stat()
                assert (after.st_size,after.st_mtime_ns)==fingerprints[str(source)],'Source changed: '+str(source)
                hashes[member]=digest.hexdigest()
            print('PACKED',name,flush=True)
        for path in [manifest_path,csv_path,Path(__file__).resolve()]:
            member='successful_landings_20261120/'+path.name
            z.write(str(path),member);hashes[member]=hashlib.sha256(path.read_bytes()).hexdigest()
        z.writestr('successful_landings_20261120/SHA256SUMS.txt',''.join(digest+'  '+name+'\n' for name,digest in sorted(hashes.items())))
    checked=0
    with zipfile.ZipFile(str(partial)) as z:
        for name,expected in hashes.items():
            digest=hashlib.sha256()
            with z.open(name) as f:
                while True:
                    chunk=f.read(1024*1024)
                    if not chunk:break
                    digest.update(chunk)
            assert digest.hexdigest()==expected,'Archive checksum mismatch: '+name
            checked+=1
        assert len(z.namelist())==len(hashes)+1
    for name,expected in fingerprints.items():
        st=Path(name).stat()
        assert (st.st_size,st.st_mtime_ns)==expected,'Original changed: '+name
    partial.rename(archive)
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        while True:
            chunk=f.read(1024*1024)
            if not chunk:break
            digest.update(chunk)
    checksum=archive.with_suffix('.zip.sha256')
    checksum.write_text(digest.hexdigest()+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,
                          included_runs=len(SELECTED),verified_entries=checked,
                          sha256=digest.hexdigest(),original_logs_preserved=True),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
