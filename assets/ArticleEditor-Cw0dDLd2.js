import{V as g}from"./index-DZ5togL0.js";import{m as v,o as p,c as u,_ as w,b as o,a as t,w as l,d as i,g as c}from"./index-B0nsGddm.js";const k={id:"vditor"},b={__name:"MarkdownEditor",setup(n){let e;window.innerWidth<768&&(e=["emoji","headings","bold","italic","strike","link","|","list","ordered-list","check","outdent","indent","|","quote","line","code","inline-code","insert-before","insert-after","|","upload","record","table","|","undo","redo","|","edit-mode","content-theme","code-theme","export",{name:"more",toolbar:["fullscreen","both","preview","info","help"]}]);const d=()=>{new g("vditor",{toolbar:e,lang:"zh_CN",mode:"wysiwyg",height:window.innerHeight-100,placeholder:"Hello, Vditor!",outline:{enable:!0,position:"right"},preview:{markdown:{toc:!0,mark:!0,footnotes:!0,autoSpace:!0},math:{engine:"KaTeX",inlineDigit:!0}},hint:{emojiPath:"https://cdn.jsdelivr.net/npm/vditor@latest/dist/images/emoji",emoji:{sd:"💔",j:"https://cdn.jsdelivr.net/npm/vditor@latest/dist/images/emoji/j.png"}},upload:{url:"/api/upload/editor",accept:"image/*",filename(r){return r.replace(/[^(a-zA-Z0-9\u4e00-\u9fa5\.)]/g,"").replace(/[\?\\/:|<>\*\[\]\(\)\$%\{\}@~]/g,"").replace("/\\s/g","")}}})};return v(()=>{d()}),(r,a)=>(p(),u("div",k))}},V={components:{editor:b},data(){return{articleTitle:"",vditor:null}},mounted(){},methods:{}},j={class:"article-editor"},C={class:"article-title"},T={class:"editor-container"},x={class:"article-actions"};function y(n,e,d,r,a,E){const m=i("a-input"),_=i("editor"),s=i("a-button"),h=i("a-space");return p(),u("div",j,[e[3]||(e[3]=o("h2",{class:"page-title"},"文章编辑",-1)),o("div",C,[t(m,{modelValue:a.articleTitle,"onUpdate:modelValue":e[0]||(e[0]=f=>a.articleTitle=f),placeholder:"Enter Article Title",size:"large"},null,8,["modelValue"])]),o("div",T,[t(_)]),o("div",x,[t(h,null,{default:l(()=>[t(s,{type:"primary",onClick:n.saveArticle},{default:l(()=>e[1]||(e[1]=[c("保存")])),_:1},8,["onClick"]),t(s,{type:"default",onClick:n.cancelEditing},{default:l(()=>e[2]||(e[2]=[c("取消")])),_:1},8,["onClick"])]),_:1})])])}const N=w(V,[["render",y],["__scopeId","data-v-aa7c2ca8"]]);export{N as default};