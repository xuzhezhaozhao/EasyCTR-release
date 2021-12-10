#ifndef ATTR_API_H
#define ATTR_API_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

/**
 * ���Ź����ڴ�д��ֵ������ֵ
 * ���ָ�������ѱ�����ֵ,��ֵ����ԭֵ
 * attr: ����ID
 * iValue: ����ֵ
 * return: -1��ʾʧ��;0��ʾ�ɹ�
 */
int Attr_API_Set(int attr,int iValue);

/**
 * ��ȡ��ֵ�����Ե�ֵ
 * ���ָ�������ѱ�����ֵ,��ֵ����ԭֵ
 * attr: ����ID
 * iValue: ����ֵ�洢��ַ
 * return: -1��ʾʧ��;0��ʾ�ɹ� 
 */
int Get_Attr_Value(int attr,int *iValue); 

/**
 * ���Ƽ�ʹ��
 * ���Ź����ڴ�д��ֵ������ֵ
 * ���ָ�������ѱ�����ֵ,��ֵ���ۼӵ�ԭֵ��
 * attr: ����ID
 * iValue: �ۼ�ֵ
 */
int Attr_API(int attr,int iValue);

/**
 * ȡδ�ò��Ź����ڴ��С
 * ����ֵ��1��ʾʧ��;������ʾδ�ù����ڴ��С
 */ 
int get_adv_memlen();

/**
 * ȡ���ò��Ź����ڴ��С
 * ����ֵ��1��ʾʧ��;������ʾ�����ù����ڴ��С
 */  
int get_adv_memusedlen();

/**
 * ���Ź����ڴ�д����,��agent���͸����ܷ���������
 * attr_id:���ݵ�����id����600��ʼ��С��600Ϊ�Ƿ�
 * len���ݳ��ȣ���ҪС�ڹ����ڴ�Ŀ��ô�С�������ڴ��ʼ���ô�С��2Mk �� sizeof(int)
 * pvalue:����ʵ��ҵ�����ݣ��ǿա�
 * ����ֵ0��ʾ�ɹ�������ʧ��
 * ����ע�⣺����������ε������ò�ͬ���ݣ����ݽ��������������У�ֱ��2M
 */ 
int adv_attr_set(int attr_id , size_t len , char* pvalue);

/**
 * ȡ���Ź����ڴ������,ע��pOut�����߷��䣬�Ҵ�Сһ��Ҫ���ڻ����len
 * offset:ƫ��������ʾ�Ӳ��Ź����ڴ濪ʼoffset���ȿ�ʼȡֵ
 * len��ȡ�����ݵĳ��ȣ�������ڹ����ڴ��С��ȡ�����ڴ���󳤶�
 * pOut:�������buffer���ɵ����߷��䣬ע���Сһ��Ҫ���ڻ����len
 */ 
int get_adv_mem(size_t offset , size_t len , char* pOut);

/**
 * ��IP�ϱ���ֵ��ҵ��������ֵ
 * strIP: �ַ���IP��ַ
 * iAttrID: ����id
 * iValue: ����ֵ
 * �ɹ�����0��ʧ�ܷ���-1
 */ 
int setNumAttrWithIP(const char* strIP, int iAttrID, int iValue);

/**
 * ��IP�ϱ��ֽ���ҵ������
 * strIP: �ַ���IP��ַ
 * iAttrID: ����id
 * len: �ֽڴ����Գ���
 * pval: �ֽڴ������׵�ַ
 * �ɹ�����0��ʧ�ܷ���-1
*/
int setStrAttrWithIP(const char* strIP, int iAttrID, size_t len , const char* pval);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif // ATTR_API_H

