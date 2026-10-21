"""Scalar data and browser regression checks. Run python -m pytest -q."""
import io
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import server
from fieldstream import encode_frame,decode_frame

@pytest.mark.parametrize('shape',[(4,3,5),(4,3,5,3)])
@pytest.mark.parametrize('dtype',['float32','int16'])
def test_frames(shape,dtype):
    a=np.linspace(-3,7,np.prod(shape),dtype=np.float32).reshape(shape)
    raw=encode_frame(a,seq=9,dtype=dtype,timestamp=1.25)
    h,b=decode_frame(raw)
    assert h['seq']==9 and h['time']==1.25
    np.testing.assert_allclose(a,b,atol=7/32767)
    with pytest.raises(ValueError):decode_frame(raw[:-1])

def test_missing():
    a=np.ones((2,2,2));a[0,0,0]=np.nan
    assert np.isnan(decode_frame(encode_frame(a))[1][0,0,0])
    with pytest.raises(ValueError):encode_frame(a,dtype='int16')

def test_rest():
    for kind in ['scalar_gaussian','scalar_potential','tornado']:
        for fmt in ['json','npy','csv','vtk','vti']:
            status,_,body,_=server.field_response(kind,{'n':'5','format':fmt})
            assert status==200
            if fmt=='npy':assert np.load(io.BytesIO(body)).shape==((5,5,5) if kind.startswith('scalar') else (5,5,5,3))

def test_browser(tmp_path):
    from playwright.sync_api import sync_playwright
    http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
    threading.Thread(target=http.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=['--enable-unsafe-swiftshader'])
            page=browser.new_page(viewport={'width':1280,'height':900})
            external=[]
            def block_external(route):
                external.append(route.request.url)
                route.abort()
            page.route('https://**/*',block_external)
            errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('console',lambda m:errors.append(m.text) if m.type=='error' and ('Shader' in m.text or 'VALIDATE' in m.text) else None)
            html=(ROOT/'index.html').read_text(encoding='utf-8')
            html=html.replace('</script>\n</body>', '''
window.app={params,FORMATS,scalarFormat,readScalarFile,makeScalar,regenerate,rebuildAll,syncMode,applyLiveFrame,setProjection,
feedFrame(raw,meta){live={meta,metaChanged:true,frames:0,refs:null,fps:0,last:0,ws:{close(){}}};applyLiveFrame(Uint8Array.from(raw).buffer);},
slicePixel(axis,sign,pos=0){
  const s=params.scalar;s.x=s.y=s.z=s.surfaces=s.points=s.voxels=s.volume=false;s[['x','y','z'][axis]]=true;
  s['p'+['x','y','z'][axis]]=pos;rebuildAll();camera.position.set(.7*sign,.8*sign,.9*sign);camera.position.setComponent(axis,sign*3+pos*field.hi[axis]);
  camera.up.set(0,axis===2?1:0,axis===2?0:1);controls.target.set(0,0,0);controls.target.setComponent(axis,pos*field.hi[axis]);controls.update();camera.lookAt(controls.target);
  renderer.render(scene,camera);const gl=renderer.getContext(),pixel=new Uint8Array(4);
  gl.readPixels(Math.floor(canvas.width/2),Math.floor(canvas.height/2),1,1,gl.RGBA,gl.UNSIGNED_BYTE,pixel);return Array.from(pixel);
},
get field(){return field;},get stats(){return scalarState;},get groups(){return groups;}};
</script>\n</body>''')
            page.route('**/index.html',lambda route:route.fulfill(body=html,content_type='text/html'))
            page.goto(f'http://127.0.0.1:{http.server_port}/index.html')
            try:page.wait_for_function('window.app && app.field',timeout=60000)
            except Exception:
                print('Browser errors:',errors)
                raise
            page.evaluate("async()=>{app.params.source='browser (JS)';app.params.mode='Scalar';app.params.resolution=12;app.syncMode();await app.regenerate();}")
            assert page.evaluate('!!app.field.scalar && app.stats.triangles>0')
            page.get_by_role('button',name='Load example: temperature',exact=True).click()
            page.wait_for_function("app.field.scalarName==='Air temperature: warm plume'")
            assert page.evaluate("app.field.dims.join(',')==='25,21,17' && app.field.spacing[2]===75 && app.field.origin[0]===-1200")
            assert page.evaluate("app.field.units===String.fromCharCode(176)+'C' && app.stats.triangles>0")

            for example in ['Atmospheric temperature','Electric dipole potential','Gyroid','Gaussian hot spot']:
                page.evaluate('(example)=>{app.params.scalar.example=example;return app.regenerate();}',example)
                assert page.evaluate('app.stats.triangles>0')
            page.screenshot(path=str(tmp_path/'slices.png'))
            for axis in range(3):
                for pos in [0,-1,1]:
                    front=page.evaluate('a=>app.slicePixel(...a)',[axis,1,pos])
                    back=page.evaluate('a=>app.slicePixel(...a)',[axis,-1,pos])
                    print('Slice axis/position',axis,pos,'front/back',front,back)
                    assert max(front[:3])>60 and max(back[:3])>60
                    assert max(abs(a-b) for a,b in zip(front,back))<=3
            page.evaluate("async()=>{app.params.source='server API';app.params.scalar.example='Atmospheric temperature';await app.regenerate();}")
            assert page.evaluate("app.field.units===String.fromCharCode(176)+'C' && app.field.scalarName==='Atmospheric temperature'")
            page.evaluate("async()=>{app.params.source='browser (JS)';app.params.scalar.example='Gaussian hot spot';await app.regenerate();}")
            for mode in ['volume','points','voxels']:
                page.evaluate("mode=>{const s=app.params.scalar;s.x=s.y=s.z=s.surfaces=s.volume=s.points=s.voxels=false;s[mode]=true;app.rebuildAll();}",mode)
                page.wait_for_timeout(250)
                assert page.evaluate('app.groups.scalar.children.length>0')
                if mode=='voxels':
                    # Every rendered voxel must be a closed box wholly inside the clipped domain.
                    for clipped in [False,True]:
                        if clipped:
                            page.evaluate("()=>{app.params.scalar.clipMinZ=-.6;app.params.scalar.clipY=.35;app.rebuildAll();}")
                        assert page.evaluate("""()=>{
                          const mesh=app.groups.scalar.children.find(o=>o.isInstancedMesh),f=app.field,s=app.params.scalar;
                          const lo=[s.clipMinX,s.clipMinY,s.clipMinZ].map((v,a)=>v*f.hi[a]);
                          const hi=[s.clipX,s.clipY,s.clipZ].map((v,a)=>v*f.hi[a]);
                          if(mesh.material.clippingPlanes?.length || mesh.geometry.index.count!==36 || !mesh.count)return false;
                          const m=mesh.instanceMatrix.array;
                          for(let i=0;i<mesh.count;i++)for(let a=0;a<3;a++){
                            const size=m[i*16+a*5],center=m[i*16+12+a];
                            if(size<=0 || center-size/2<lo[a]-1e-6 || center+size/2>hi[a]+1e-6)return false;
                          }return true;
                        }""")
                    page.evaluate("()=>{app.params.scalar.clipMinZ=-1;app.params.scalar.clipY=1;app.rebuildAll();}")
                if mode=='volume':
                    page.screenshot(path=str(tmp_path/'volume.png'))
                    print('Volume screenshot:',tmp_path/'volume.png')
                    page.evaluate("()=>{app.setProjection('orthographic');app.params.scalar.projection='Maximum';app.rebuildAll();}")
                    page.wait_for_timeout(200)
                    page.screenshot(path=str(tmp_path/'projection.png'))
                    page.evaluate("()=>app.setProjection('perspective')")
            report=page.evaluate('''async()=>{
              const results=[];
              for(const name of Object.keys(app.FORMATS)){
                const f=app.field,buf=await new Blob(app.scalarFormat(name).write(f)).arrayBuffer();
                const r=app.readScalarFile(name,buf)||app.FORMATS[name].read(buf);
                results.push({name,scalar:r.scalar,n:r.data.length,error:Math.max(...r.data.map((v,i)=>Math.abs(v-f.scalar[i])))});
              }return results;
            }''')
            assert all(r['scalar'] and r['n']==12**3 and r['error']<1e-5 for r in report),report
            assert page.evaluate('''()=>{const a=app.makeScalar({dims:[2,2,2],data:[0,-2,NaN,1,2,3,4,5]});return a.scalar[0]===0&&a.scalar[1]===-2&&a.mask[2]===1;}''')
            page.evaluate("async()=>{app.params.mode='Vector';app.params.generator='Tornado';app.syncMode();await app.regenerate();}")
            assert page.evaluate('!app.field.scalar && app.groups.lines.children.length>0')
            assert page.get_by_role('button',name='Load example: terrain wind',exact=True).count()==1
            for shape in [(3,4,5),(3,4,5,3)]:
                raw=encode_frame(np.ones(shape,dtype=np.float32))
                page.evaluate("raw=>app.feedFrame(raw,{name:'Test stream',units:'K'})",list(raw))
                assert page.evaluate('!!app.field.scalar')==(len(shape)==3)
            page.evaluate("async()=>{app.params.source='browser (JS)';app.params.mode='Vector';await app.regenerate();}")
            # Exercise the real import control with non-cubic, Fortran-order scalar data.
            source=np.asfortranarray(np.linspace(-4,7,60).reshape(3,4,5))
            source[1,1,1]=np.nan
            payload=io.BytesIO();np.save(payload,source)
            page.locator('#fileInput').set_input_files({'name':'temperature.npy','mimeType':'application/octet-stream','buffer':payload.getvalue()})
            page.wait_for_function("app.params.mode==='Scalar' && app.field.dims.join(',')==='5,4,3'")
            assert page.evaluate('app.field.scalar[0]===-4 && app.field.mask.reduce((a,b)=>a+b,0)===1')
            # Bad CSV data must not silently become a volume with duplicate/missing cells.
            assert page.evaluate("()=>{try{app.readScalarFile('CSV (.csv)',new TextEncoder().encode('x,y,z,value\\n0,0,0,1\\n0,0,0,2').buffer);return false;}catch{return true;}}")
            assert not external,external
            # Direct file opening retains the CDN path; HTTP hosting above is fully offline.
            page.unroute('https://**/*',block_external)

            page.goto((ROOT/'index.html').as_uri(),wait_until='domcontentloaded')
            page.wait_for_function("document.getElementById('genName').textContent==='Tornado'",timeout=60000)
            assert page.locator('#genName').inner_text()=='Tornado'
            assert not errors,errors
            browser.close()
    finally:
        http.shutdown();http.server_close()


def test_stream_server_vendor():
    import stream_server
    with stream_server.app.test_client() as client:
        for path in ['three/build/three.module.js','three/examples/jsm/controls/OrbitControls.js','three/examples/jsm/utils/BufferGeometryUtils.js','lil-gui/lil-gui.esm.js']:
            response=client.get('/vendor/'+path)
            assert response.status_code==200
            assert 'javascript' in response.content_type
            response.close()
